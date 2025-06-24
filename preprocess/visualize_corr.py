
import os
import sys
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import imageio.v2 as imageio
import cv2
from PIL import Image
from numpy import random


def draw_corr(rgbA, rgbB, corrA, corrB, output_name):
    vis = np.concatenate([rgbA, rgbB], axis=1)
    radius = 2
    for i in range(len(corrA)):
        uvA = corrA[i]
        uvB = corrB[i].copy()
        uvB[0] += rgbA.shape[1]
        color = tuple(np.random.randint(0, 255, size=(3)).tolist())
        vis = cv2.circle(vis, uvA, radius=radius, color=color, thickness=1)
        vis = cv2.circle(vis, uvB, radius=radius, color=color, thickness=1)
        vis = cv2.line(vis, uvA, uvB, color=color, thickness=1, lineType=cv2.LINE_AA)
    imageio.imwrite(f'{output_name}.png', vis.astype(np.uint8))

def read_img_dict(folder, name, poses=None):
    # rgb_subfolder = 'color_raw'
    # if not os.path.exists(pjoin(folder, rgb_subfolder)):
    # rgb_subfolder = 'color_segmented'
    img_dict = {}
    # for key, subfolder in (('rgb', rgb_subfolder)):
    img_name = pjoin(folder, 'color_segmented', f'{name}.png')
    img = np.array(Image.open(img_name))
    # if key == 'mask' and len(img.shape) == 3:
    #     img = img[..., 0]
    img_dict['rgb'] = img
    # if poses is None:
    #     poses = yaml.safe_load(open(pjoin(folder, 'init_keyframes.yml'), 'r'))
    # img_dict['cam2world'] = list_to_array(poses[f'frame_{name}']['cam_in_ob']).reshape(4, 4)

    return img_dict

data_dir= "/workspace/tmp_dataset/artnerf/data/multi_part/StorageFurniture" # 
vis_path = 'outputs_loftr_7'
os.makedirs(vis_path,exist_ok=True)
corr_path = pjoin(data_dir, 'correspondence_loftr/no_filter')
# import pdb;
# pdb.set_trace()
if os.path.exists(corr_path):
    for filename in tqdm(os.listdir(corr_path)):
        if filename.endswith('npz'):
            cur_corr = np.load(pjoin(corr_path, filename), allow_pickle=True)['data']
            for corr in cur_corr:
                for order in [1, -1]:
                    src_name, tgt_name = list(corr.keys())#[::order]
                    src_pixel, tgt_pixel = corr[src_name], corr[tgt_name]
                    src_dict = read_img_dict(data_dir, src_name)
                    tgt_dict = read_img_dict(data_dir, tgt_name)
                    # import pdb;pdb.set_trace()
                    x=random.randint(100, size=(len(src_pixel)))
                    left_mask = x>70
                    new_src_pixel= src_pixel[left_mask]
                    new_tgt_pixel= tgt_pixel[left_mask]
                    print("remaining {} pixels from total".format(len(new_src_pixel)),len(src_pixel))
                    draw_corr(src_dict['rgb'],tgt_dict['rgb'],  src_pixel, tgt_pixel, pjoin(vis_path, f'{src_name}_{tgt_name}'))
                    draw_corr(src_dict['rgb'],tgt_dict['rgb'],  new_src_pixel, new_tgt_pixel, pjoin(vis_path, f'{src_name}_{tgt_name}_filter'))
                    # import pdb 
                    # pdb.set_trace()




            # for dic_i in range(len(cur_corr)):
            #     for keys in cur_corr[dic_i]:
            #         # keys_i = cur_corr[dic_i].keys()
            #         coor_i= cur_corr[keys]
            #         import pdb 
            #         pdb.set_trace()
            #         src_dict = read_img_dict(data_dir, keys_i[0])
            #         tgt_dict = read_img_dict(data_dir, keys_i[1])
            #         draw_corr(src_dict['rgb'],tgt_dict['rgb'], coor_i[0], coor_i[1], pjoin(vis_path, f'{keys_i[0]}_{keys_i[1]}'))
            #         import pdb 
            #         pdb.set_trace()

'''
CUDA_VISIBLE_DEVICES=0 python visualize_corr.py --data_path /workspace/tmp_dataset/artnerf/data/multi_part/StorageFurniture --output_path results/multi_part/StorageFurniture/correspondence_loftr 

'''