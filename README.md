# ReMark (LampMark + Arc2Face Wrapper)

This repo contains exactly two folders:
- `LampMark/` (your cloned repo)
- `arc2face_wrapper/` (portable Arc2Face expression/ref adapter wrapper)

You can copy `arc2face_wrapper/` into another project and use the provided config to run generation without CLI args.

## Quick Start

1) Prepare environment
- Activate the existing conda env (same one you already use for Arc2Face):
```
conda activate arc2face
```

2) Update config
- Edit `arc2face_wrapper/config/generate_config.json` and set:
  - `models_dir` to the folder that contains your Arc2Face models
  - `manifest` to your CSV triplet list
  - `output_dir` to where you want results

3) Run example generator
```
cd /path/to/ReMark
LIBS=$(printf ':%s' /home/ldy/miniconda3/envs/arc2face/lib/python3.10/site-packages/nvidia/*/lib); LIBS=${LIBS:1}
PYTHONNOUSERSITE=1 HF_HOME=/path/to/.hf_cache LD_LIBRARY_PATH="$LIBS:${LD_LIBRARY_PATH:-}" \
/home/ldy/miniconda3/envs/arc2face/bin/python arc2face_wrapper/scripts/generate_example.py
```

## Configuration Reference

File: `arc2face_wrapper/config/generate_config.json`

- `models_dir` (string): Path to Arc2Face `models/` directory  
- `manifest` (string): Path to CSV with columns `source_image`, `expression_image`, `reference_image`  
- `output_dir` (string): Output directory  
- `limit` (int): How many rows to process  
- `start_index` (int): Row offset  
- `strict_cuda_provider` (bool): If `true`, error out if onnxruntime CUDA provider is unavailable  

`config` block (generation settings):
- `use_ref_adapter` (bool): Enable Reference Adapter  
- `lora_ref_scale` (float): Reference strength  
- `num_steps` (int): Diffusion steps  
- `guidance_scale` (float): Guidance scale  
- `num_images` (int): Images per input  
- `exp_adapter_scale` (float): Expression adapter strength  
- `output_size` (int): Output resolution (must be multiple of 8)  
- `seed` (int|null): Seed for determinism  

## Notes

- Models are loaded from `models_dir`. This folder is expected to already contain:
  - `arc2face/`
  - `encoder/`
  - `exp_adapter/`
  - `ref_adapter/`
  - `smirk/`
  - `antelopev2/`
  If you copy `models/` from an existing Arc2Face setup, you do not need to download anything again.
- `PYTHONNOUSERSITE=1` is recommended to avoid conflicts with `~/.local` packages.
- If you move this folder to a new machine or project, you only need to update paths in `arc2face_wrapper/config/generate_config.json`.
