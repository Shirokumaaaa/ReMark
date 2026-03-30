import glob
import pandas as pd
from PIL import Image
from natsort import natsorted
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import config as c

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class Test_Dataset(Dataset):
    def __init__(self, path, format):
        self.transform = T.Compose([
            T.Resize(128),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.files = natsorted(sorted(glob.glob(path + "/*." + format)))

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index])
            image = to_rgb(image)
            item = self.transform(image)
            return item
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.files)

class INN_Dataset(Dataset):
    def __init__(self, transforms, mode="train"):
        self.transform = transforms
        if mode == 'train':
            df = pd.read_csv(c.TRAIN_CSV)
        else:
            df = pd.read_csv(c.VAL_CSV)
        self.files = df['img_path'].tolist()

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index])
            image = to_rgb(image)
            item = self.transform(image)
            return item
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.files)

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.Resize([c.cropsize, c.cropsize]),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

transform_val = T.Compose([
    T.Resize([c.cropsize_val, c.cropsize_val]),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

trainloader = DataLoader(
    INN_Dataset(transforms=transform, mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    drop_last=True
)
# Test Dataset loader
testloader = DataLoader(
    INN_Dataset(transforms=transform_val, mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)





