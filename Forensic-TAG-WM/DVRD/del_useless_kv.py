import torch

checkpoint_path = '/public/chenyuzhuo/MODELS/image_watermarking_models/TAG-WM/DVRD/checkpoints/trainsize-512_epochnum-100_totalstep-33400.pt'
save_path = '/public/chenyuzhuo/MODELS/image_watermarking_models/TAG-WM/DVRD/checkpoints/clean-trainsize-512_epochnum-100_totalstep-33400.pt'

checkpoint = torch.load(checkpoint_path)
print("Original keys:", checkpoint.keys())
keys = list(checkpoint.keys())
for key in keys:
    if key != 'model_state_dict':
        del checkpoint[key]
print("Processed keys:", checkpoint.keys())
torch.save(checkpoint, save_path)