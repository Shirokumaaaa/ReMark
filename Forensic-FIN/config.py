import os

epochs = 500
clamp = 2.0

# optimizer
# Two-stage training schedule (warmup -> main).
warmup_epochs = 15
warmup_lr = 1e-4
lr = 1e-3
betas = (0.5, 0.999)
gamma = 0.5
weight_decay = 1e-5

noise_flag = True

# input settings
warmup_message_weight = 10000
message_weight = 100
warmup_stego_weight = 1
stego_weight = 1
message_length = 64

# Train:
batch_size = 50
cropsize = 128

# Val:
batchsize_val = 16
cropsize_val = 128

# Data Manifests (CSV pointing to Dataset-CelebA_HQ)
_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(_ROOT, 'data_manifests', 'celeba_hq_128_800_train.csv')
VAL_CSV   = os.path.join(_ROOT, 'data_manifests', 'celeba_hq_128_100_val.csv')
TEST_CSV  = os.path.join(_ROOT, 'data_manifests', 'celeba_hq_128_test.csv')

# Saving checkpoints:
MODEL_PATH = os.path.join(_ROOT, 'experiments', 'celeba_hq_128') + os.sep
SAVE_freq = 5

suffix = ''
train_continue = False
diff = False






