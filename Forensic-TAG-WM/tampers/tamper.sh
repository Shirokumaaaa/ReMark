#!/bin/bash

#SBATCH -N 1                                        # number of nodes
#SBATCH -p gpu5                                     # partition
#SBATCH -c 8                                        # number of cpus
#SBATCH --mem 30G                                    # memory
#SBATCH --gres gpu:1                                # number of gpus of each node
#SBATCH --mail-user=yz.chen@mail.sdu.edu.cn
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --output %j-%x.log                          # %x is job-name, %j is job ID

# #SBATCH -o logs/%j.sleep                          # standerd output file
# #SBATCH -e logs/%j.sleep                          # error output file

# env: gaussianshading
# conda deactivate
# conda activate gaussianshading

cd /public/chenyuzhuo/models/image_watermarking_models/Gaussian-Shading   # SLURM_SUBMIT_DIR
echo "job begin"

# python -m tampers.random_crop
# python -m tampers.random_drop
# python -m tampers.logo_putting.logo_putting
python -m tampers.logo_putting.random_logo_putting


echo "job end"


