"""
MIT License

Copyright (c) 2024 Mohamed El Banani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as transforms
import yaml
# from .utils import read_image
from tqdm import tqdm
from PIL import Image, ImageOps

def read_image(image_path: str, exif_transpose: bool = True) -> Image.Image:
    """Reads a NAVI image (and rotates it according to the metadata)."""
    with open(image_path, "rb") as f:
        with Image.open(f) as image:
            if exif_transpose:
                image = ImageOps.exif_transpose(image)
            image.convert("RGB")
            return image


def make_tf(translation_cvcam, sc_factor):
        tf = np.eye(4)
        tf[:3, 3] = translation_cvcam
        tf1 = np.eye(4)
        tf1[:3, :3] *= sc_factor
        tf = tf1 @ tf
        return tf

glcam_in_cvcam = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])

def list_to_array(l):
    if isinstance(l, list):
        return np.stack([list_to_array(x) for x in l], axis=0)
    elif isinstance(l, str):
        return np.array(float(l))
    elif isinstance(l, float):
        return np.array(l)
    
def read_img_dict(folder, name, poses=None):
    rgb_subfolder = 'color_raw'
    if not os.path.exists(os.path.join(folder, rgb_subfolder)):
        rgb_subfolder = 'color_segmented'
    img_dict = dict(img_name=f'{name}.png') # {}
    # for key, subfolder in (('rgb', rgb_subfolder), ('mask', 'mask')):
    #     img_name = os.path.join(folder, subfolder, f'{name}.png')
    #     img = np.array(Image.open(img_name))
    #     if key == 'mask' and len(img.shape) == 3:
    #         img = img[..., 0]
    #     img_dict[key] = img
    if poses is None:
        poses = yaml.safe_load(open(os.path.join(folder, 'init_keyframes.yml'), 'r'))
    img_dict['cam2world'] = list_to_array(poses[f'frame_{name}']['cam_in_ob']).reshape(4, 4)

    return img_dict, poses

class ObjectPairsDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        # Some defaults for consistency.
        self.name = "StorageFurniture"
        self.root = f"/workspace/tmp_dataset/artnerf/data/multi_part/{self.name}"
        self.split = "test"
        self.num_views = 2

        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((800, 800)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        # parse files for data
       
        self.scene_info0 = np.load(f'{self.root}/start/scene_normalization.npz', allow_pickle=True)
        self.trans0 = make_tf(self.scene_info0['translation'],self.scene_info0['sc_factor'])
        # Print out dataset stats
        self.scene_info1 = np.load(f'{self.root}/end/scene_normalization.npz', allow_pickle=True)
        self.trans1 = make_tf(self.scene_info1['translation'],self.scene_info1['sc_factor'])
        # Print out dataset stats
        self.top_k= 30 #-1
        self.num_tgt_per_src= -1
        self.instances = self.get_instances(self.root)
        self.K = np.loadtxt(f"{self.root}/cam_K.txt").reshape(3, 3)
        print(f"{self.name} | {len(self.instances)} pairs")

    def get_dep(self, path):
        with open(path, "rb") as f:
            with Image.open(f) as img:
                img = np.array(img)
                img = torch.tensor(img).float() / 1000.0
                return img[None, :, :]

    def get_instances(self, root_path):
        corr_path=f"{root_path}/correspondence_dino"
        save_name = f'tgt_all'
        if self.top_k >= 0:
            save_name = f'{save_name}_top{self.top_k}'
            top_k=self.top_k
        

        os.makedirs(os.path.join(corr_path,'no_filter'),exist_ok=True)

        if os.path.exists(os.path.join(corr_path,f"test_{self.top_k}.npz")):
            print(f"loading from {root_path}/test.npz")
            instances = np.load(os.path.join(corr_path,f"test_{self.top_k}.npz"),allow_pickle= True)["name"]
            if self.top_k < 0:
                top_k= np.sqrt(len(instances))
        
        else:
            all_frames = yaml.safe_load(open(os.path.join(root_path, 'init_keyframes.yml'), 'r'))
            src_names = ['_'.join(frame_name.split('_')[-2:]) for frame_name in all_frames]
           
            instances = []
            ins_dict=dict(name=[])
            pbar = tqdm(src_names)
            top_k = self.top_k
            for src_name in pbar:
                # src_instances = []
                pbar.set_description(src_name)
                src_time = int(src_name.split('_')[0])
                src_dict, poses = read_img_dict(root_path, src_name)
                src_position = src_dict['cam2world'][:3, 3]

                all_tgt = []
                for frame_name in poses:
                    if poses[frame_name]['time'] != src_time:
                        tgt_position = list_to_array(poses[frame_name]['cam_in_ob']).reshape(4, 4)[:3, 3]
                        all_tgt.append(('_'.join(frame_name.split('_')[-2:]), ((src_position - tgt_position) ** 2).sum()))
                all_tgt.sort(key=lambda x: x[1])
                if self.top_k <= 0:
                    top_k = len(all_tgt)
                tgt_names = [pair[0] for pair in all_tgt[:top_k]]
                # tgt_dicts = [read_img_dict(root_path, tgt_name, poses)[0] for tgt_name in tgt_names]
                for tgt_name in tgt_names:
                    tgt_dict = read_img_dict(root_path, tgt_name, poses)[0]
                    # instances.append((src_dict, tgt_dict, K))
                    instances.append({src_name: src_dict['cam2world'], tgt_name: tgt_dict['cam2world']})
                    ins_dict['name'].append({src_name: src_dict['cam2world'], tgt_name: tgt_dict['cam2world']})
                    # src_instances.append({src_name: src_dict['cam2world'], tgt_name: tgt_dict['cam2world']})
             
                # np.savez_compressed(os.path.join(corr_path,'no_filter',f"src_{src_name}_{save_name}.npz") ,name=src_instances)
   
            np.savez_compressed(os.path.join(corr_path,f"test_{self.top_k}.npz"),name=ins_dict['name'])
        self.num_tgt_per_src = top_k
        return instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        ins_dic= self.instances[index]
        src_name, tgt_name=ins_dic.keys()
        src_time = int(src_name.split('_')[0])
        # paths
        # rgb_path_0 = os.path.join(self.root, f"color/{ins_0}.jpg")
        # rgb_path_1 = os.path.join(self.root, s_id, f"color/{ins_1}.jpg")
        # dep_path_0 = os.path.join(self.root, s_id, f"depth/{ins_0}.png")
        # dep_path_1 = os.path.join(self.root, s_id, f"depth/{ins_1}.png")

        # get rgb
        rgb_0o = read_image(os.path.join(self.root,'color_segmented',src_name+'.png'), exif_transpose=False)
        rgb_1o = read_image(os.path.join(self.root,'color_segmented',tgt_name+'.png'), exif_transpose=False) #read_image(rgb_path_1, exif_transpose=False)
      
        rgb_0 = self.rgb_transform(rgb_0o)
        rgb_1 = self.rgb_transform(rgb_1o)

        # get depths
        dep_0 = self.get_dep(os.path.join(self.root,'depth_filtered',src_name+'.png'))
        dep_1 = self.get_dep(os.path.join(self.root,'depth_filtered',tgt_name+'.png'))

        # get poses
        # pose_path_0 = os.path.join(self.root, s_id, f"pose/{ins_0}.txt")
        # pose_path_1 = os.path.join(self.root, s_id, f"pose/{ins_1}.txt")
        # Rt_0 = torch.tensor(ins_dic[src_name] @ glcam_in_cvcam)

        # Rt_1 = torch.tensor(ins_dic[tgt_name] @ glcam_in_cvcam)

        if src_time==0:
            trans0= self.trans0
            trans1=self.trans1
        else:
            trans0= self.trans1
            trans1=self.trans0
        Rt_0 = torch.tensor(trans0 @ (ins_dic[src_name] @ glcam_in_cvcam)) #torch.tensor(ins_dic[src_name] ) #
        Rt_1 = torch.tensor(trans1 @ (ins_dic[tgt_name] @ glcam_in_cvcam)) #torch.tensor(ins_dic[tgt_name] ) # 
        Rt_01 = Rt_1.inverse() @ Rt_0
        # import pdb
        # pdb.set_trace()
        return {
            "uid": index,
            "class_id": "Object_test",
            "sequence_id": self.name,
            "src_name": src_name,
            "tgt_name": tgt_name,
            "K": torch.tensor(self.K).float(),
            "rgb_0": rgb_0,
            "rgb_1": rgb_1,
            "rgb_0o": np.array(rgb_0o.resize((800,800))),
            "rgb_1o": np.array(rgb_1o.resize((800,800))),
            "depth_0": dep_0,
            "depth_1": dep_1,
            "Rt_0": torch.eye(4).float(),
            "Rt_1": Rt_01.float(),
        }
