import argparse
import copy
import csv
import cv2
import json
import numpy as np
import os
import ruamel.yaml
from sklearn.cluster import DBSCAN
import kaolin
import open3d as o3d
import joblib
from PIL import Image
import imageio.v2 as imageio
from os.path import join as pjoin
yaml = ruamel.yaml.YAML()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def find_biggest_cluster(pts, eps=0.005, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    dbscan.fit(pts)
    ids, cnts = np.unique(dbscan.labels_, return_counts=True)
    best_id = ids[cnts.argsort()[-1]]
    keep_mask = dbscan.labels_ == best_id
    pts_cluster = pts[keep_mask]
    return pts_cluster, keep_mask

def compute_translation_scales(pts, max_dim=2, cluster=True, eps=0.005, min_samples=5):
    if cluster:
        pts, keep_mask = find_biggest_cluster(pts, eps, min_samples)
    else:
        keep_mask = np.ones((len(pts)), dtype=bool)
    max_xyz = pts.max(axis=0)
    min_xyz = pts.min(axis=0)
    center = (max_xyz + min_xyz) / 2
    sc_factor = max_dim / (max_xyz - min_xyz).max()  # Normalize to [-1,1]
    sc_factor *= 0.9
    translation_cvcam = -center
    return translation_cvcam, sc_factor, keep_mask


glcam_in_cvcam = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])


def toOpen3dCloud(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def depth2xyzmap(depth, K):
    invalid_mask = (depth < 0.1)
    H, W = depth.shape[:2]
    vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
    zs = depth.reshape(-1)
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = pts.reshape(H, W, 3).astype(np.float32)
    xyz_map[invalid_mask] = 0
    return xyz_map.astype(np.float32)

def compute_scene_bounds_worker(color_file, K, glcam_in_world, use_mask, rgb=None, depth=None, mask=None):
    if rgb is None:
        depth_file = color_file.replace('images', 'depth_filtered')
        mask_file = color_file.replace('images', 'masks')
        rgb = np.array(Image.open(color_file))[..., :3]
        depth = cv2.imread(depth_file, -1) / 1e3
    xyz_map = depth2xyzmap(depth, K)
    valid = depth >= 0.1
    if use_mask:
        if mask is None:
            mask = cv2.imread(mask_file, -1)
        valid = valid & (mask > 0)
    pts = xyz_map[valid].reshape(-1, 3)
    if len(pts) == 0:
        return None
    colors = rgb[valid].reshape(-1, 3)

    pcd = toOpen3dCloud(pts, colors)

    pcd = pcd.voxel_down_sample(0.01)
    new_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    cam_in_world = glcam_in_world @ glcam_in_cvcam
    new_pcd.transform(cam_in_world)

    return np.asarray(new_pcd.points).copy(), np.asarray(new_pcd.colors).copy()

BAD_DEPTH = 99
BAD_COLOR = 128
def mask_and_normalize_data(rgbs, depths, masks, poses, sc_factor, translation):
    '''
    @rgbs: np array (N,H,W,3)
    @depths: (N,H,W)
    @masks: (N,H,W)
    @normal_maps: (N,H,W,3)
    @poses: (N,4,4)
    '''
    depths[depths < 0.1] = BAD_DEPTH
    if masks is not None:
        rgbs[masks == 0] = BAD_COLOR
        depths[masks == 0] = BAD_DEPTH
        masks = masks[..., None]

    rgbs = (rgbs / 255.0).astype(np.float32)
    depths *= sc_factor
    depths = depths[..., None]
    poses[:, :3, 3] += translation
    poses[:, :3, 3] *= sc_factor

    return rgbs, depths, masks, poses

def compute_scene_bounds(color_files, glcam_in_worlds, K, use_mask=True, base_dir=None, rgbs=None, depths=None,
                         masks=None, cluster=True, translation_cvcam=None, sc_factor=None, eps=0.06, min_samples=1):
    # assert color_files is None or rgbs is None

    if base_dir is None:
        base_dir = os.path.dirname(color_files[0]) + '/../'
    os.makedirs(base_dir, exist_ok=True)

    args = []
    if rgbs is not None:
        for i in range(len(rgbs)):
            args.append((color_files[i], K, glcam_in_worlds[i], use_mask, rgbs[i], depths[i], masks[i]))
    else:
        for i in range(len(color_files)):
            args.append((color_files[i], K, glcam_in_worlds[i], use_mask))

    print(f"compute_scene_bounds_worker start")
    ret = joblib.Parallel(n_jobs=6, prefer="threads")(joblib.delayed(compute_scene_bounds_worker)(*arg) for arg in args)
    print(f"compute_scene_bounds_worker done")

    pcd_all = None
    for r in ret:
        if r is None or len(r[0]) == 0:
            continue
        if pcd_all is None:
            pcd_all = toOpen3dCloud(r[0], r[1])
        else:
            pcd_all += toOpen3dCloud(r[0], r[1])

    pcd = pcd_all.voxel_down_sample(eps / 5)

    pts = np.asarray(pcd.points).copy()

    def make_tf(translation_cvcam, sc_factor):
        tf = np.eye(4)
        tf[:3, 3] = translation_cvcam
        tf1 = np.eye(4)
        tf1[:3, :3] *= sc_factor
        tf = tf1 @ tf
        return tf

    if translation_cvcam is None:
        translation_cvcam, sc_factor, keep_mask = compute_translation_scales(pts, cluster=cluster, eps=eps,
                                                                             min_samples=min_samples)
        tf = make_tf(translation_cvcam, sc_factor)
    else:
        tf = make_tf(translation_cvcam, sc_factor)
        tmp = copy.deepcopy(pcd)
        tmp.transform(tf)
        tmp_pts = np.asarray(tmp.points)
        keep_mask = (np.abs(tmp_pts) < 1).all(axis=-1)

    print(f"compute_translation_scales done")

    pcd = toOpen3dCloud(pts[keep_mask], np.asarray(pcd.colors)[keep_mask])
    pcd_real_scale = copy.deepcopy(pcd)

    with open(f'{base_dir}/normalization.yml', 'w') as ff:
        tmp = {
            'translation_cvcam': translation_cvcam.tolist(),
            'sc_factor': float(sc_factor),
        }
        yaml.dump(tmp, ff)

  
    pcd.transform(tf)
   
    return sc_factor, translation_cvcam, pcd_real_scale, pcd


parser = argparse.ArgumentParser(
    description=
    "Run neural graphics primitives testbed with additional configuration & output options"
)

parser.add_argument("--scene_folder", type=str, default="project/tmp_dataset/paris/scissor_11100")
parser.add_argument("--save_dir", type=str, default="project/tmp_dataset/paris/scissor_11100")
# parser.add_argument("--scaled_image", action="store_true")
# parser.add_argument("--semantics", action="store_true")
args = parser.parse_args()
basedir = args.scene_folder

print(f"processing folder: {basedir}")

# Step for generating training images
step = 1
data_dir=args.scene_folder

mode= 'end'
save_dir = pjoin(args.save_dir, mode)
K = np.loadtxt(pjoin(data_dir, 'cam_K.txt')).reshape(3, 3)

keyframes = yaml.load(open(pjoin(data_dir, 'init_keyframes.yml'), 'r'))
keys = list(keyframes.keys())

frame_ids = []
timesteps = []
cam_in_obs = []
if mode=='start':
    new_keys=[k for k in keys if 'frame_00000' in k]
if mode=='end':
    new_keys=[k for k in keys if 'frame_00001' in k]
for k in new_keys:
   
    cam_in_ob = np.array(keyframes[k]['cam_in_ob']).reshape(4, 4)
    cam_in_obs.append(cam_in_ob)
    timesteps.append(float(keyframes[k]['time']))
    frame_ids.append(k.replace('frame_', ''))
cam_in_obs = np.array(cam_in_obs)
timesteps = np.array(timesteps)

max_timestep = np.max(timesteps) + 1
normalized_timesteps = timesteps / max_timestep

# import pdb 
# pdb.set_trace()
frame_names, rgbs, depths, masks = [], [], [], []

rgb_dir = pjoin(data_dir, mode,'images')
for frame_id in frame_ids:
    rgb_file = pjoin(rgb_dir, f'{frame_id}.png')

    rgb = imageio.imread(rgb_file)
    rgb_wh = rgb.shape[:2]
    depth = cv2.imread(rgb_file.replace('images', 'depth'), -1) / 1e3
    depth_wh = depth.shape[:2]
    if rgb_wh[0] != depth_wh[0] or rgb_wh[1] != depth_wh[1]:
        depth = cv2.resize(depth, (rgb_wh[1], rgb_wh[0]), interpolation=cv2.INTER_NEAREST)

    mask = cv2.imread(rgb_file.replace('images', 'mask'), -1)
    if len(mask.shape) == 3:
        mask = mask[..., 0]

    frame_names.append(rgb_file)
    rgbs.append(rgb)
    depths.append(depth)
    masks.append(mask)

glcam_in_obs = cam_in_obs

scene_normalization_path = pjoin(data_dir,mode, 'scene_normalization.npz')
if os.path.exists(scene_normalization_path):
    scene_info = np.load(scene_normalization_path, allow_pickle=True)
    sc_factor, translation = scene_info['sc_factor'], scene_info['translation']
    pcd_normalized = scene_info['pcd_normalized']
    pcd_normalized = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_normalized.astype(np.float64)))
else:
    sc_factor, translation, pcd_real_scale, pcd_normalized = compute_scene_bounds(frame_names, glcam_in_obs, K,
                                                                                    use_mask=True,
                                                                                    base_dir=save_dir,
                                                                                    rgbs=np.array(rgbs),
                                                                                    depths=np.array(depths),
                                                                                    masks=np.array(masks),
                                                                                    cluster=False, eps=0.01,
                                                                                    min_samples=5,
                                                                                    sc_factor=None,
                                                                                    translation_cvcam=None)
    np.savez_compressed(scene_normalization_path, sc_factor=sc_factor, translation=translation,
                        pcd_normalized=np.asarray(pcd_normalized.points),pcd_color=np.asarray(pcd_normalized.colors))

print("sc factor", sc_factor, 'translation', translation)

rgbs, depths, masks, poses = mask_and_normalize_data(np.array(rgbs), depths=np.array(depths),
                                                        masks=np.array(masks),
                                                        poses=glcam_in_obs,
                                                        sc_factor=sc_factor,
                                                        translation=translation)
# intrinsic_file = os.path.join(basedir, "intrinsic/intrinsic_color.txt")
# import pdb 
# pdb.set_trace()
intrinsic = K
print("intrinsic parameters:")
print(intrinsic)
np.savetxt(os.path.join(data_dir,mode,'intrinsics.txt'), K)
print(f"total number of training frames: {len(frame_names)}")
# print(f"total number of testing frames: {len(test_ids)}")
W, H =rgb.shape[:2]
# train_ids=np.arange(len(rgbs))
# os.makedirs(os.path.join(data_dir, 'color_scaled'), exist_ok=True)
# times = np.unique(timesteps)

transform_json = {}
transform_json["fl_x"] = K[0, 0]
transform_json["fl_y"] = K[1, 1]
transform_json["cx"] = K[0, 2]
transform_json["cy"] = K[1, 2]
transform_json["w"] = W
transform_json["h"] = H
transform_json["camera_angle_x"] = np.arctan2(W / 2, K[0, 0]) * 2
transform_json["camera_angle_y"] = np.arctan2(H / 2, K[1, 1]) * 2
transform_json["aabb_scale"] = 16
transform_json["frames"] = []

time_all =np.unique(timesteps)
for t in time_all:

    def write_into_single_file(split,frame_ids):
        file_cont_train = dict(frames=[])
        pose_folder = rgb_dir.replace('images', 'pose')
        os.makedirs(pose_folder,exist_ok = True)
        save_pose =True
        if split=='train':
            n_step=1
        if split == 'test':
            n_step =10
            save_pose=False
        for idx,frame_id in enumerate(frame_ids):
            ####################
            index = int(idx)
            ####################
            rgb_file= pjoin('images', f'{frame_id}.png')
            depth_file=rgb_file.replace('images', 'depth')
            # label_file=label_list[idx].replace(data_dir+'/','')
            # object_file=sammask_list[idx].replace(data_dir+'/','')
            pose_file = pjoin(pose_folder, f'{frame_id}.txt')

            pose = poses[index]
            cur_dict= {
                'file_path': rgb_file,
                'depth': depth_file,
                # 'label': label_file,
                # 'object':object_file,
                'transform_matrix': pose,
            }
            if idx % n_step==0:
                file_cont_train['frames'].append(cur_dict)
            # save single pose txt into pose folder
            if save_pose:
                np.savetxt(pose_file, pose)
            # import pdb
            # pdb.set_trace()

        file_cont_train.update({
            "fl_x": K[0, 0],
            "fl_y": K[1, 1],
            "cx": K[0, 2],
            "cy": K[1, 2],
            "w": W,
            "h": H,
            "camera_angle_x": np.arctan2(W / 2, K[0, 0]) * 2,
            "camera_angle_y": np.arctan2(H / 2, K[1, 1]) * 2,
            "aabb_scale": 16
        }
        )

        # save transform.json in each scene
        out_file=os.path.join(data_dir,mode,'transforms_{}.json'.format(split))
        with open(out_file,'w') as f:
            json.dump(file_cont_train,f,cls=NpEncoder,indent='\t')

    cur_mask= timesteps-t==0
    # import pdb
    # pdb.set_trace()
    # write_into_file(str(t),[frame_ids[ids] for ids in np.where(cur_mask)[0]])
    write_into_single_file('train',[frame_ids[ids] for ids in np.where(cur_mask)[0]])
    write_into_single_file('test',[frame_ids[ids] for ids in np.where(cur_mask)[0]])
    # file_name = 'transforms_train_'+str(t)
    # file_name += ".json"
    # out_file = open(pjoin(data_dir,file_name), "w")
    # json.dump(transform_json, out_file, indent=4)
    # out_file.close()
