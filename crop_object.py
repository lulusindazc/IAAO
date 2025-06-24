import os
import numpy as np
from PIL import Image
import cv2
import glob

dataset_name = 'ramen' #'sofa'
# texts = ["red bag", "black leather shoe", "banana", "hand", "camera", "white sheet"]

# dataset_name = 'sofa'
# texts = ["Pikachu", "a stack of UNO cards", "a red Nintendo Switch joy-con controller", "Gundam", "Xbox wireless controller", "grey sofa"]
# dataset_name = 'ramen'
# texts = ["chopsticks", "egg", "glass of water", "pork belly", "wavy noodles in bowl", "yellow bowl"]

# com_type = 
# image_path=''
# mask_path= ''


ouptut_dir =  '..'+  dataset_name 
eval_name = "2D_masks/semantic-sam-5" # "sam_clip_features"
# rgb_name= "images"
gt_images_pth = f"{ouptut_dir}/{eval_name}"
# pred_images_pth = f"{ouptut_dir}/{eval_name}/pred_images"
pred_segs_pth = f"{ouptut_dir}/{eval_name}/pred_segs"
# rele_pth = f"{ouptut_dir}/{eval_name}/relevancy"
os.makedirs(pred_segs_pth,exist_ok =True)

# img_list= glob.glob(os.path.join(gt_images_pth,'*_mask.png')) #os.listdir(gt_images_pth)

# img_list= [os.path.join(gt_images_pth,'00_mask.png'),os.path.join(gt_images_pth,'01_mask.png')]
img_list = [os.path.join(gt_images_pth,'frame_00001_mask.png'), os.path.join(gt_images_pth,'frame_00002_mask.png')]
def to_transparent_bg(img_tmp):
    datas = img_tmp.getdata()
    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img_tmp.putdata(newData)
    return img_tmp

for img_p in img_list:
    image_name= os.path.basename(img_p).replace('_mask.png','')
    # img= np.array(Image.open(img_p))
    # import pdb; pdb.set_trace()
    rgb_image = cv2.imread(os.path.join(f"{ouptut_dir}/images",image_name+'.jpg'))
    
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    img_h,img_w= rgb_image.shape[0],rgb_image.shape[1]
    segmentation_image = cv2.imread(img_p, cv2.IMREAD_UNCHANGED)[:,:,0]
    segmentation_image = cv2.resize(segmentation_image.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    ids = np.unique(segmentation_image)
    for txt in ids:
        img_out= np.zeros_like(rgb_image)
        # import pdb; pdb.set_trace()
        if os.path.exists(f"{pred_segs_pth}/{image_name}/{txt}.png"):
            continue
        else:
            os.makedirs(f"{pred_segs_pth}/{image_name}",exist_ok =True)
       
        mask = (segmentation_image==txt)
        img_out[mask]=rgb_image[mask]
        img_t= Image.fromarray(img_out)
        img_t = img_t.convert('RGBA')
        img_t=to_transparent_bg(img_t)

        img_t.save(f"{pred_segs_pth}/{image_name}/{txt}_img.png") #, **png_info)
            
        # import pdb; pdb.set_trace()

