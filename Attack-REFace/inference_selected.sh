
# Set variables
name="faceswap_outputs"
Results_dir="data/${name}/results"
Base_dir="data/${name}/Outs"
Results_out="data/${name}/results/results" 
device=0


CONFIG="models/REFace/configs/project_ffhq.yaml"
CKPT="models/REFace/checkpoints/last.ckpt"


#change this
target_path="data/faceswap_inputs/target"
source_path="data/faceswap_inputs/source"




# Run inference
# ideal for small number of samples

CUDA_VISIBLE_DEVICES=${device} python scripts/inference_swap_selected.py \
    --outdir "${Results_dir}" \
    --target_folder "${target_path}" \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --src_folder "${source_path}" \
    --Base_dir "${Base_dir}" \
    --n_samples 1 \
    --scale 3.5 \
    --ddim_steps 50



