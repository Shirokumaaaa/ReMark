import torch
from torchvision import transforms
import torch.nn as nn
import math
import torch.nn.functional as F

def from_pretrained(checkpoint_path, train_size=512, torch_dtype=torch.float16, strict=True, device='cpu'):
    train_size = train_size // 8  # downsample 8x for VAE encoded images
    model = TrainableDVRD(train_size=train_size, torch_dtype=torch_dtype, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("DVRD state dict loaded:", model.unet.load_state_dict(checkpoint['model_state_dict'], strict=strict))
    model.eval()
    return model

class TrainableDVRD(nn.Module):
    '''Dense Variation Region Detector'''
    def __init__(self,
                 train_size=512,
                 torch_dtype=torch.float16,
                 device='cpu'):
        super(TrainableDVRD, self).__init__()
        if train_size is not None:
            self.transform = transforms.Compose([transforms.Resize(train_size)])
        else:
            self.transform = None
        self.unet = UNet(in_channels=4, out_channels=4).to(device)
        if torch_dtype == torch.float16:
            self.unet = self.unet.half()
            
    def forward(self, x):
        if self.transform:
            h, w = x.size()[-2:]
            x = self.transform(x)
            
        x = self.unet(x)
        
        if self.transform:
            x = transforms.Resize((h, w))(x)
            
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        self.encoder5 = conv_block(512, 1024)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.upconv4 = up_conv(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = up_conv(512, 256)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = up_conv(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = up_conv(128, 64)
        self.decoder1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))
        
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)

class TrainfreeDVRD(torch.nn.Module):
    def __init__(self, 
                 max_kernel_size = 8,
                 adaptive_max_kernel_size = True,
                 overlapping = False,
                 ):
        super(TrainfreeDVRD, self).__init__()
        self.max_kernel_size = max_kernel_size
        self.adaptive_max_kernel_size = adaptive_max_kernel_size
        self.overlapping = overlapping
        
    def forward(self, change_img: torch.Tensor, confidence = 0.5):
        """
        input: 
            changed_img: [B, 1, H, W], img with changes
            confidence: a value to control the confidence of change detection
        return: 
            [B, 1, H, W]
        """
        if change_img.dim() == 3:
            change_img = change_img.unsqueeze(0)
            dim = 3
        change_img = change_img.float()
        addition_times = 1
        print('change_img(1):', change_img)

        # adaptively adjust the max_kernel_size
        if self.adaptive_max_kernel_size:
            # The max_kernel_size should be proportional to the image size.
            self.max_kernel_size = int(8 * (change_img.size(2) * change_img.size(3)) / (64 * 64))

        if self.overlapping:
            for k in range(3, self.max_kernel_size + 1, 2):
                change_img += F.avg_pool2d(change_img, kernel_size=k, stride=1, padding=k // 2)
                addition_times += 1
        else:
            for k in range(1, int(math.log2(self.max_kernel_size)) + 1):
                k = 2 ** k
                print('k:', k)
                temp_change = F.avg_pool2d(change_img, kernel_size=k, stride=k, padding=0)
                temp_change = torch.repeat_interleave(temp_change, repeats=k, dim=2)
                temp_change = torch.repeat_interleave(temp_change, repeats=k, dim=3)
                change_img += temp_change
                addition_times += 1
                
        change_img = change_img / addition_times
        
        # Apply the change confidence threshold
        change_img = (change_img >= confidence).int()
        print(f'change_img({addition_times}):', change_img)

        if dim == 3:
            change_img = change_img[0]
        return change_img