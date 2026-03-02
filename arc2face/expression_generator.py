import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel
from insightface.app import FaceAnalysis

from arc2face import CLIPTextModelWrapper, ReferenceAdapter, image_align, project_face_embs
from arc2face.exp_utils import ExpressionEncoder, run_smirk


ImageInput = Union[str, Image.Image, np.ndarray]


@dataclass
class ExpressionGenerationConfig:
    use_ref_adapter: bool = True
    lora_ref_scale: float = 1.0
    num_steps: int = 25
    guidance_scale: float = 3.0
    num_images: int = 1
    exp_adapter_scale: float = 1.0
    output_size: int = 256
    seed: Optional[int] = None


class Arc2FaceExpressionGenerator:
    def __init__(
        self,
        models_dir: str = "models",
        base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        device: Optional[str] = None,
        strict_cuda_provider: bool = True,
    ) -> None:
        self.models_dir = models_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._lock = threading.Lock()
        self.strict_cuda_provider = strict_cuda_provider

        if self.device == "cuda":
            providers = ort.get_available_providers()
            has_cuda_provider = "CUDAExecutionProvider" in providers
            if strict_cuda_provider and not has_cuda_provider:
                raise RuntimeError(
                    "CUDA device requested but onnxruntime CUDA provider is unavailable. "
                    "Check CUDA runtime libraries and LD_LIBRARY_PATH."
                )
            if not has_cuda_provider:
                print(
                    "[Warning] CUDA requested, but onnxruntime CUDA provider is unavailable. "
                    "InsightFace will fall back to CPU."
                )

        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0

        self.app = FaceAnalysis(name="antelopev2", root="./", providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(256, 256))
        self.fa = None

        encoder = CLIPTextModelWrapper.from_pretrained(
            self.models_dir, subfolder="encoder", torch_dtype=self.dtype
        )
        unet = UNet2DConditionModel.from_pretrained(
            self.models_dir, subfolder="arc2face", torch_dtype=self.dtype
        )
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            base_model,
            text_encoder=encoder,
            unet=unet,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config
        )
        self.pipeline = self.pipeline.to(self.device)

        self.pipeline.load_ip_adapter(
            self.models_dir,
            subfolder="exp_adapter",
            weight_name="exp_adapter.bin",
            image_encoder_folder=None,
        )
        self.pipeline.load_lora_weights(
            f"{self.models_dir}/ref_adapter",
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="ref",
        )

        ref_unet = UNet2DConditionModel.from_pretrained(
            self.models_dir, subfolder="arc2face", torch_dtype=self.dtype
        ).to(self.device)
        self.ref_adapter_w = ReferenceAdapter(ref_unet, mode="write")
        self.ref_adapter_r = ReferenceAdapter(self.pipeline.unet, mode="read", cfg=True)

        self.smirk_encoder = ExpressionEncoder(n_exp=50).to(self.device)
        checkpoint = torch.load(
            f"{self.models_dir}/smirk/SMIRK_em1.pt", map_location=self.device
        )
        checkpoint_encoder = {
            k.replace("smirk_encoder.expression_encoder.", ""): v
            for k, v in checkpoint.items()
            if "smirk_encoder.expression_encoder." in k
        }
        self.smirk_encoder.load_state_dict(checkpoint_encoder)
        self.smirk_encoder.eval()

    @staticmethod
    def _load_image(inp: ImageInput) -> Image.Image:
        if isinstance(inp, Image.Image):
            return inp.convert("RGB")
        if isinstance(inp, str):
            return Image.open(inp).convert("RGB")
        if isinstance(inp, np.ndarray):
            if inp.ndim != 3:
                raise ValueError("Input numpy image must be HxWxC.")
            if inp.dtype != np.uint8:
                inp = np.clip(inp, 0, 255).astype(np.uint8)
            return Image.fromarray(inp).convert("RGB")
        raise TypeError("Unsupported image input type.")

    def _prepare_source_image(self, img: Image.Image, use_ref_adapter: bool, output_size: int) -> Image.Image:
        if not use_ref_adapter:
            return img
        if self.fa is None:
            import face_alignment
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, flip_input=False, device=self.device
            )

        face_landmarks, _, bboxes = self.fa.get_landmarks(np.array(img), return_bboxes=True)
        if face_landmarks is None:
            raise ValueError("Face detection failed on source image.")

        if len(face_landmarks) > 1:
            sizes = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
            lmks = face_landmarks[int(np.argmax(sizes))]
        else:
            lmks = face_landmarks[0]

        return image_align(img, lmks, output_size=output_size)

    def _get_id_embedding(self, img: Image.Image) -> torch.Tensor:
        faces = self.app.get(np.array(img)[:, :, ::-1])
        if len(faces) == 0:
            raise ValueError("Face detection failed on source image.")
        faces = sorted(
            faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1])
        )[-1]
        id_emb = torch.tensor(faces["embedding"], dtype=self.dtype)[None].to(self.device)
        id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)
        return project_face_embs(self.pipeline, id_emb)

    def _get_expression_embeddings(self, exp_img: Image.Image, num_images: int) -> torch.Tensor:
        outputs = run_smirk(self.smirk_encoder, np.array(exp_img), device=self.device)
        if outputs is None:
            raise ValueError("Face detection failed on expression image.")
        exp_embs = torch.cat(
            [outputs["expression_params"], outputs["eyelid_params"], outputs["jaw_params"]], dim=1
        ).to(dtype=self.dtype)
        exp_adapter_embeds = torch.cat([torch.zeros_like(exp_embs[:, None, :]), exp_embs[:, None, :]], dim=0)
        return exp_adapter_embeds.repeat_interleave(repeats=num_images, dim=0)

    def _prepare_reference_adapter(self, img: Image.Image, id_emb: torch.Tensor, num_images: int) -> None:
        ref_img = (torch.tensor(np.array(img), dtype=self.dtype).to(self.device).permute(2, 0, 1) / 255) * 2 - 1
        ref_img = torch.stack([ref_img, ref_img]).repeat_interleave(repeats=num_images, dim=0)
        ref_img = self.pipeline.vae.encode(ref_img).latent_dist.sample()
        ref_img = ref_img * self.pipeline.vae.config.scaling_factor
        encoder_hidden_states = torch.cat([id_emb, id_emb], dim=0).repeat_interleave(repeats=num_images, dim=0)

        self.ref_adapter_w.unet(
            ref_img,
            torch.zeros(ref_img.size(0), device=ref_img.device).long(),
            encoder_hidden_states,
            return_dict=False,
        )
        self.ref_adapter_r.update(self.ref_adapter_w)

    def generate(
        self,
        source_image: ImageInput,
        expression_image: ImageInput,
        reference_image: Optional[ImageInput] = None,
        config: Optional[ExpressionGenerationConfig] = None,
    ) -> List[Image.Image]:
        cfg = config or ExpressionGenerationConfig()
        if cfg.output_size <= 0 or cfg.output_size % 8 != 0:
            raise ValueError("output_size must be a positive multiple of 8.")
        seed = cfg.seed if cfg.seed is not None else int(torch.randint(0, 2**31 - 1, (1,)).item())

        src_img = self._load_image(source_image)
        exp_img = self._load_image(expression_image)
        ref_img = self._load_image(reference_image) if reference_image is not None else src_img

        with self._lock:
            self.pipeline.set_ip_adapter_scale(cfg.exp_adapter_scale)
            self.pipeline.set_adapters("ref", cfg.lora_ref_scale if cfg.use_ref_adapter else 0.0)

            ref_img = self._prepare_source_image(ref_img, cfg.use_ref_adapter, cfg.output_size)
            id_emb = self._get_id_embedding(src_img)
            exp_adapter_embeds = self._get_expression_embeddings(exp_img, cfg.num_images)

            generator = torch.Generator(device=self.device).manual_seed(seed)

            try:
                if cfg.use_ref_adapter:
                    self._prepare_reference_adapter(ref_img, id_emb, cfg.num_images)

                images = self.pipeline(
                    prompt_embeds=id_emb.repeat_interleave(repeats=cfg.num_images, dim=0),
                    ip_adapter_image_embeds=[exp_adapter_embeds],
                    num_inference_steps=cfg.num_steps,
                    guidance_scale=cfg.guidance_scale,
                    height=cfg.output_size,
                    width=cfg.output_size,
                    generator=generator,
                ).images
            finally:
                # Always clear stateful adapters to avoid contamination across samples.
                self.ref_adapter_r.clear()
                self.ref_adapter_w.clear()

        return images

    def generate_batch(
        self,
        pairs: List[Dict[str, ImageInput]],
        config: Optional[ExpressionGenerationConfig] = None,
        stop_on_error: bool = False,
    ) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        cfg = config or ExpressionGenerationConfig()
        for idx, pair in enumerate(pairs):
            src = pair.get("source_image")
            exp = pair.get("expression_image")
            ref = pair.get("reference_image")
            if src is None or exp is None:
                msg = "Each batch item must contain source_image and expression_image."
                if stop_on_error:
                    raise ValueError(msg)
                results.append({"index": idx, "ok": False, "error": msg, "images": []})
                continue

            try:
                images = self.generate(src, exp, reference_image=ref, config=cfg)
                results.append({"index": idx, "ok": True, "error": None, "images": images})
            except Exception as exc:  # noqa: BLE001
                if stop_on_error:
                    raise
                results.append({"index": idx, "ok": False, "error": str(exc), "images": []})
        return results
