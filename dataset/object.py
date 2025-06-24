import open3d as o3d
import numpy as np
import os
import cv2
from evaluation.constants import SCANNET_LABELS, SCANNET_IDS
import ruamel.yaml
yaml = ruamel.yaml.YAML()

def make_tf(translation_cvcam, sc_factor):
        tf = np.eye(4)
        tf[:3, 3] = translation_cvcam
        tf1 = np.eye(4)
        tf1[:3, :3] *= sc_factor
        tf = tf1 @ tf
        return tf


class ObjectDataset:

    def __init__(self, seq_name,obj_name) -> None:
        self.seq_name = seq_name
        self.obj_name= obj_name #'StorageFurniture' 'Refrigerator'
        # self.base_root= f"/workspace/tmp_dataset/paris/{self.obj_name}" #f'/workspace/tmp_dataset/artnerf/data/multi_part/{self.obj_name}'
        self.base_root= f'/workspace/tmp_dataset/artnerf/data/multi_part/{self.obj_name}'
        self.root = f'{self.base_root}/{seq_name}'
        self.rgb_dir = f'{self.root}/images'
        self.depth_dir = f'{self.root}/depth'
        self.segmentation_dir = f'{self.root}/sam_clip_features' #part_mask'
        self.object_dict_dir = f'{self.root}/mask'
        self.point_cloud_path =  f'{self.root}/scene_normalization.npz' #f'{self.root}/{seq_name}_vh_clean_2.ply' 
        self.mesh_path = self.point_cloud_path
        self.extrinsics_dir = f'{self.base_root}/init_keyframes.yml' ##f'{self.root}/pose'
        
        self.keyframes = yaml.load(open(self.extrinsics_dir, 'r'))
        
        self.scene_info = np.load(self.point_cloud_path, allow_pickle=True)
        # sc_factor, translation = self.scene_info['sc_factor'], self.scene_info['translation']
        self.trans = make_tf(self.scene_info['translation'],self.scene_info['sc_factor'])
        self.depth_scale = 1000.0
        self.image_size = (800, 800)
    

    def get_frame_list(self, stride):
        image_list = os.listdir(self.rgb_dir)
        image_list = sorted(image_list, key=lambda x: int(x.split('.')[0]))
        # import pdb
        # pdb.set_trace()
        end = int(image_list[-1].split('.')[0]) + 1
        frame_id_list = [img_n.split('.')[0] for img_n in image_list] #np.arange(0, end, stride)
        return frame_id_list
    

    def get_intrinsics(self, frame_id):
        intrinsic_path = f'{self.root}/intrinsics.txt'
        intrinsics = np.loadtxt(intrinsic_path)

        intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
        intrinisc_cam_parameters.set_intrinsics(800, 800, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
        return intrinisc_cam_parameters
    

    def get_extrinsic(self, frame_id):
        # pose_path = os.path.join(self.extrinsics_dir, str(frame_id) + '.txt')
        # pose = np.loadtxt(pose_path)
        pose= np.array(self.keyframes[f'frame_{frame_id}']['cam_in_ob']).reshape(4, 4)
        glcam_in_cvcam = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])
        cam_in_world = pose @ glcam_in_cvcam
        # import pdb
        # pdb.set_trace()
        cam_tran=cam_in_world 
        return cam_tran
    

    def get_depth(self, frame_id, align_with_depth=True):
        depth_path = os.path.join(self.depth_dir, str(frame_id) + '.png')
        depth = cv2.imread(depth_path, -1)
        depth = depth / self.depth_scale
        depth = depth.astype(np.float32)

        mask_path = os.path.join(self.object_dict_dir , f'{frame_id}.png')
        if not os.path.exists(mask_path):
            assert False, f"Segmentation not found: {mask_path}"
        obj_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if align_with_depth:
            obj_mask = cv2.resize(obj_mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        depth[obj_mask<1]=0
        return depth


    def get_rgb(self, frame_id, change_color=True):
        rgb_path = os.path.join(self.rgb_dir, str(frame_id) + '.png')
        rgb = cv2.imread(rgb_path)

        if change_color:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb    


    def get_segmentation(self, frame_id, align_with_depth=False):
        segmentation_path = os.path.join(self.segmentation_dir, f'{frame_id}.png')
        # import pdb; pdb.set_trace()
        if not os.path.exists(segmentation_path):
            assert False, f"Segmentation not found: {segmentation_path}"
        segmentation = cv2.imread(segmentation_path, cv2.IMREAD_UNCHANGED)
        return segmentation

    def get_frame_path(self, frame_id):
        rgb_path = os.path.join(self.rgb_dir, str(frame_id) + '.png')
        segmentation_path = os.path.join(self.segmentation_dir, f'{frame_id}.png')
        return rgb_path, segmentation_path
    

    def get_label_features(self):
        label_features_dict = np.load(f'data/text_features/object.npy', allow_pickle=True).item()
        return label_features_dict


    def get_scene_points(self, out_colors=False):
        # mesh = o3d.io.read_triangle_mesh(self.point_cloud_path)
        # vertices = np.array(mesh.vertices)
        # scene_info = np.load(self.point_cloud_path, allow_pickle=True)
       
        pcd_normalized = self.scene_info['pcd_normalized']
        # pcd_normalized = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_normalized.astype(np.float64)))
        # vertices = np.asarray(pcd_normalized.points)
        if out_colors:
            colors = self.scene_info['pcd_color']
            return pcd_normalized, colors
        else:
            return pcd_normalized
    
   
    def get_label_id(self):
        self.class_id = SCANNET_IDS
        self.class_label = SCANNET_LABELS

        self.label2id = {}
        self.id2label = {}
        for label, id in zip(self.class_label, self.class_id):
            self.label2id[label] = id
            self.id2label[id] = label

        return self.label2id, self.id2label