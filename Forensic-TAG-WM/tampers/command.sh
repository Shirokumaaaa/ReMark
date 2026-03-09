### kill all opened tensorboard
kill -9 $(ps -ef|grep tensorboard|grep -v grep|awk '{print $2}')

### change the proxy to the initial default setting
export https_proxy=http://127.0.0.1:7890

### change the source
-i https://pypi.mirrors.ustc.edu.cn/simple/

# test logo_putting
srun python -m tampers.logo_putting.logo_putting

##### tampers
### basic settings
cd /public/chenyuzhuo/models/image_watermarking_models/Gaussian-Shading
salloc -N 1 -p gpu3 -c 8 --mem 30G --gres gpu:1
conda deactivate

# random_crop
srun python -m tampers.random_crop