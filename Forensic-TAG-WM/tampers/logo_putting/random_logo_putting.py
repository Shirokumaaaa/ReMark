import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import os.path as p
from PIL import Image
from torchvision import transforms
import random

def CheckPositions(check, positions):
    for i in positions:
        if (check[0]>=(i[0]+i[2]+20) or check[0]+check[2] <=i[0]-20) or (check[1]>=(i[1]+i[3]+20) or check[1]+check[3] <=i[1]-20):
                continue
        else:
            return False
    return True

def GetBoundingBox_Image(image, mask):
    mask[mask>0]=1
    mask[mask<1]=0
    c_masks = mask.permute(1,0,2,3)
    # print(c_masks.shape)
    c_masks.squeeze_(0)
    mask_ids = torch.unique(c_masks)[1:]
    masks = c_masks == mask_ids[:,None,None]
    # print(masks.shape)
    boxes = torchvision.ops.masks_to_boxes(masks)
    boxes = boxes.int()
    x1,y1,x2,y2 = boxes[0]
    # print(boxes)
    cut_img = image * mask
    input_img = cut_img[:,:,y1:y2,x1:x2]
    cut_mask = mask[:,:,y1:y2,x1:x2]
    return input_img, cut_mask

class Logo(nn.Module):
    def __init__(self, args):
        super(Logo, self).__init__()
        #
        self.trans = transforms.ToTensor()
        self.device = args.device
        self.logopath = args.logo_data_path
        self.logonum = args.logo_putting_num
        self.logoratio = args.logo_ratio
        self.logopath_path = p.join(self.logopath,"images","%s.jpg")
        self.maskpath_path = p.join(self.logopath,"masks","%s.png")
        self.im_msk_ids = []
        for file in os.listdir(self.logopath+"/images"):
            self.im_msk_ids.append(file.split(".")[0])

    def get_logo_and_mask(self, logonum):
        indices = np.random.randint(0, len(self.im_msk_ids), size=logonum)
        l = []
        for i in indices:
            im_id = self.im_msk_ids[i]
            img = Image.open(self.logopath_path%im_id)
            mask = Image.open(self.maskpath_path%im_id)

            img = self.trans(img)
            mask = self.trans(mask)
            img = img * 2. - 1.
            img = img.to(self.device)
            mask = mask.to(self.device)
            l.append((img.unsqueeze_(0),mask.unsqueeze_(0)))
        return l
    
    def logo_cover(self, cover_img, logos:list, logo_ratio=0.2):
        # logos:[(logo,mask)]
        hb,wb = cover_img.shape[2:]
        postions = []
        for i in range(0,len(logos)):
            flag = 0
            h_i, w_i = int(hb * logo_ratio), int(wb * logo_ratio)
            logo_size = (h_i, w_i)
            y = np.random.randint(0,hb-h_i)
            x = np.random.randint(0,wb-w_i)

            num = 0
            while not flag:
                if len(postions) == 0:
                    break
                while not CheckPositions([x,y,w_i,h_i],postions):
                    y = np.random.randint(0,hb-h_i)
                    x = np.random.randint(0,wb-w_i)
                    num += 1
                    if num > 100:
                        return None, None, True
                flag = 1

                
            postions.append([x,y,w_i,h_i])
        
        merge_img = cover_img.to(self.device)
        final_masks = torch.zeros((1,1,merge_img.shape[2],merge_img.shape[3])).to(self.device)
        for i, logoitem in enumerate(logos):
            logo = logoitem[0]
            logo_mask = logoitem[1]
            input_logo_image, cut_logo_mask = GetBoundingBox_Image(logo,logo_mask)
            input_logo_image = F.interpolate(input_logo_image,size=logo_size)
            cut_logo_mask = F.interpolate(cut_logo_mask,size=logo_size)
            postion = postions[i]
            pad_mask = F.pad(cut_logo_mask , (postion[0] , wb-postion[0]-postion[2] , postion[1] , hb-postion[1]-postion[3]),value=0)
            pad_cut_img = F.pad(input_logo_image , (postion[0] , wb-postion[0]-postion[2] , postion[1] , hb-postion[1]-postion[3]),value=0)
            merge_img = merge_img * (1 - pad_mask) + pad_cut_img
            final_masks += pad_mask

        return merge_img, final_masks, False

    def forward(self, image):
        freezed = True
        while freezed:
            logos = self.get_logo_and_mask(self.logonum)
            logoed_image, gt, freezed = self.logo_cover(image, logos, logo_ratio=self.logoratio)
        return logoed_image, gt

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)     # logo putting
    parser.add_argument('--logo_putting_num', default=5, type=int)     # logo putting
    parser.add_argument('--logo_ratio', default=0.5, type=float)
    parser.add_argument('--logo_data_path', default='/public/chenyuzhuo/models/image_watermarking_models/TAG-WM/tampers/logo_putting/SOIM/train', type=str)  
    # parser.add_argument('--logo_data_path', default='./SOIM/train/', type=str)  
    args = parser.parse_args()
    ### debug code
    # img = Image.open('./tampers/logo_putting/SOIM/backgrounds/Au_ani_00001.jpg')
    # img.save('./tampers/logo_putting/test_code_results/Au_ani_00001.jpg')
    # img = transforms.ToTensor()(img).unsqueeze(0)
    # logo_instance = Logo(args)
    # # Apply logo distortion
    # img_tensor, tamper_loc = logo_instance(img)
    # # Convert tensor back to PIL image
    # img = Image.fromarray(((img_tensor.squeeze(0).permute(1, 2, 0)).cpu().numpy() * 255).astype(np.uint8))
    # print(tamper_loc)
    # print(tamper_loc.shape)
    # tamper_loc = Image.fromarray((tamper_loc.squeeze(0).permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy() * 255).astype(np.uint8))
    # img.save('./tampers/logo_putting/test_code_results/Au_ani_00001_tampered.jpg')
    # tamper_loc.save('./tampers/logo_putting/test_code_results/Au_ani_00001_gt.jpg')

    ### using
    def process_images(input_path, img_output_path, mask_output_path, logo_putting_nums, logo_putting_ratios, second_name):
        print(f'start {second_name}')
        num = 0
        num_and_ratios = list(zip(logo_putting_nums, logo_putting_ratios))

        image_names = os.listdir(input_path)
        sorted_image_names = sorted(image_names, key=lambda x: int(x.split('.')[0].split('_')[1]))
        for i, image_name in enumerate(sorted_image_names):
            if image_name.endswith('.png') is False:
                continue
            image_path = os.path.join(input_path, image_name)
            image = Image.open(image_path)
            if image is None:
                continue
            image = transforms.ToTensor()(image).unsqueeze(0)
            # Apply logo distortion
            args.logo_putting_num, args.logo_ratio = random.choice(num_and_ratios)
            logo_instance = Logo(args)
            img_tensor, tamper_loc = logo_instance(image)
            # Convert tensor back to PIL image
            image = Image.fromarray(((img_tensor.squeeze(0).permute(1, 2, 0)).cpu().numpy() * 255).astype(np.uint8))
            mask = Image.fromarray((tamper_loc.squeeze(0).permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy() * 255).astype(np.uint8))
            
            output_image_path = os.path.join(img_output_path, image_name)
            output_mask_path = os.path.join(mask_output_path, str(i).zfill(4) + '.png')
            
            image.save(output_image_path)
            mask.save(output_mask_path)
                
            print(f'save tampered img to {output_image_path}')
            print(f'save mask to {output_mask_path}')

            num += 1
            print('putting_num = ', num)
    # basic settings
    # root_dir = '/public/chenyuzhuo/models/image_watermarking_models/TAG-WM/datasets/RTLO_dataset/TLT_center50/Stable-Diffusion-Prompts/Stable-Diffusion-2-1-base/inference_step50'
    # root_dir = '/public/chenyuzhuo/models/image_watermarking_models/TAG-WM/datasets/RTLO_dataset/TLT_intervals4/Stable-Diffusion-Prompts/Stable-Diffusion-2-1-base/inference_step50'
    root_dir = '/public/chenyuzhuo/models/image_watermarking_models/TAG-WM/datasets/RTLO_dataset/TLT_center50/Stable-Diffusion-Prompts/Stable-Diffusion-2-1-base/test_set/inference_step50'
    # second_names = ['DDIM', 'DEIS', 'DPMSolver', 'PNDM', 'UniPC']
    second_names = ['DDIM']
    # second_names = ['DEIS']
    # second_names = ['DPMSolver']
    # second_names = ['PNDM']
    # second_names = ['UniPC']
    # third_names = ['512x512', '512x1024', '768x768', '768x1024', '1024x1024']
    third_names = ['512x512']
    fourth_input_name = 'images'
    # fourth_output_img_name = 'tampered_images/LogoPutting'
    # fourth_out_put_mask_name = 'tamper_loc_masks/LogoPutting'
    fourth_output_img_name = 'images_LogoPutting'
    output_mask_dir = '/public/chenyuzhuo/models/image_watermarking_models/TAG-WM/datasets/random_logo_masks_512'

    for second_name in second_names:
        for third_name in third_names:
            input_path = os.path.join(root_dir, second_name, third_name, fourth_input_name)
            img_output_path = os.path.join(root_dir, second_name, third_name, fourth_output_img_name)
            mask_output_path = output_mask_dir
            os.makedirs(img_output_path, exist_ok=True)
            os.makedirs(mask_output_path, exist_ok=True)
            # settings for random crop
            logo_putting_nums = [1, 3, 5, 7, 9]
            logo_putting_ratios = [0.7, 0.39, 0.25, 0.2, 0.1]
            
            process_images(input_path, img_output_path, mask_output_path, logo_putting_nums, logo_putting_ratios, second_name)