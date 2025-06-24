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
from datetime import datetime
import os
import hydra
import numpy as np
import torch
import torch.nn.functional as nn_F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import imageio
import cv2
from dataset.object_pairs import ObjectPairsDataset
from utils.correspondence import (
    compute_binned_performance,
    estimate_correspondence_depth,
    project_3dto2d,
)
from utils.transformations import so3_rotation_angle, transform_points_Rt



def draw_corr(rgbA, rgbB, corrA, corrB, output_name):
    vis = np.concatenate([rgbA, rgbB], axis=1).copy()
    radius = 2
    for i in range(len(corrA)):
        uvA = corrA[i]
        uvB = corrB[i].copy()
        # import pdb 
        # pdb.set_trace()
        uvB[0] += rgbA.shape[1]
        color = tuple(np.random.randint(0, 255, size=(3)).tolist())
        vis = cv2.circle(vis, uvA, radius=radius, color=color, thickness=1)
        vis = cv2.circle(vis, uvB, radius=radius, color=color, thickness=1)
        vis = cv2.line(vis, uvA, uvB, color=color, thickness=1, lineType=cv2.LINE_AA)
    # import pdb; pdb.set_trace()
    imageio.imwrite(f'{output_name}.png', vis.astype(np.uint8))


def get_image_paths(directory):
    image_paths = []
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_image_extensions):
                image_paths.append(file) #os.path.join(root, file))

    return sorted(image_paths)


@hydra.main("./configs", "object_correspondence", None)
def main(cfg: DictConfig):
    print(f"Config: \n {OmegaConf.to_yaml(cfg)}")
    dataset = ObjectPairsDataset()
    loader = DataLoader(
        dataset, 8, num_workers=4, drop_last=False, pin_memory=True, shuffle=False
    )
    # dataname = "StorageFurniture"
    # dataroot = f"/workspace/tmp_dataset/artnerf/data/multi_part/{dataname}"
    cache_file = os.path.join(dataset.root,'color_segmented','dino_descriptors_stride2.pt')
    # mix_cache_path =os.path.join(dataset.root, "color_segmented/pyramid_0.05/cache", "level_0.npy")
    if os.path.exists(cache_file):
        print(f"[DinoExtractor] Trying to load ...")
        dino_feats = torch.load(cache_file)
        print(f"[DinoExtractor] Loaded ... Feature shape is {dino_feats.shape}")
        # clip_feats = torch.from_numpy(np.load(mix_cache_path)).float()
        # print(f"[ClipExtractor] Loaded ... Feature shape is {clip_feats.shape}")
        image_paths = get_image_paths(os.path.join(dataset.root,'color_segmented'))
        dino_feat_dict=dict()
        clip_feat_dict=dict()
        for i in range(len(image_paths)):
            dino_feat_dict[image_paths[i].split('.')[0]]=dino_feats[i]# (C,H,W)
            # clip_feat_dict[image_paths[i].split('.')[0]]=clip_feats[i].permute(2,0,1) # (C,H,W)
            # import pdb; pdb.set_trace()

    else:
        # ===== Get model and dataset ====
        model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
        model = model.to("cuda")
    

    # extract features
    err_2d = []
    R_gt = []

    visualize=True
    filter_level_list=['no_filter']
    filtered_corr = {key: [] for key in filter_level_list}

    save_name = f'tgt_all'
    if dataset.top_k >= 0:
        save_name = f'{save_name}_top{dataset.top_k}'
    # count_n=0
    # num_src= len(dataset)/dataset.num_tgt_per_src
    for i in tqdm(range(len(dataset))):
        
        instance = dataset.__getitem__(i)
        rgbs = torch.stack((instance["rgb_0"], instance["rgb_1"]), dim=0)
        deps = torch.stack((instance["depth_0"], instance["depth_1"]), dim=0)
        src_name =instance['src_name']
        tgt_name= instance['tgt_name']
        K_mat = instance["K"].clone()
        Rt_gt = instance["Rt_1"].float()[:3, :4]
        R_gt.append(Rt_gt[:3, :3])
        
        h,w=instance["rgb_0"].shape[1],instance["rgb_0"].shape[2]
        if os.path.exists(cache_file):
            src_feat= dino_feat_dict[src_name]
            tgt_feat= dino_feat_dict[tgt_name]
            # sampled_clips_src = nn_F.interpolate(clip_feat_dict[src_name][None,], size=(h,w), mode='nearest')
            # sampled_dinos_src = nn_F.interpolate(dino_feat_dict[src_name][None,], size=(h, w), mode='bilinear', align_corners=False)
            # # import pdb; pdb.set_trace()
            # src_feat=torch.cat((sampled_clips_src,sampled_dinos_src),dim=1).squeeze(0) #clip_feat_dict[src_name], dino_feat_dict[src_name]
            # sampled_clips_tgt = nn_F.interpolate(clip_feat_dict[tgt_name][None,], size=(h,w), mode='nearest')
            # sampled_dinos_tgt = nn_F.interpolate(dino_feat_dict[tgt_name][None,], size=(h, w), mode='bilinear', align_corners=False)
            # tgt_feat= torch.cat((sampled_clips_tgt,sampled_dinos_tgt),dim=1).squeeze(0)#clip_feat_dict[tgt_name] #dino_feat_dict[tgt_name]
        else:
            feats = model(rgbs.cuda())
            if cfg.multilayer:
                feats = torch.cat(feats, dim=1)
            # scale depth and intrinsics
            feats = feats.detach().cpu()
            src_feat=feats[0]
            tgt_feat=feats[1]
        deps = nn_F.interpolate(deps, scale_factor=cfg.scale_factor)
        K_mat[:2, :] *= cfg.scale_factor

        # compute corr
        corr_xyz0, corr_xyz1, corr_dist = estimate_correspondence_depth(
           src_feat , tgt_feat, deps[0], deps[1], K_mat.clone(), cfg.num_corr
        )
        # valid_mask= corr_dist>=0.05
        # corr_xyz0=corr_xyz0[valid_mask]
        # corr_xyz1=corr_xyz1[valid_mask]
        # print(corr_dist)
        print("src_name:{},tgt_name:{}".format(src_name,tgt_name))
        # import pdb 
        # pdb.set_trace()
        ################ compute error
        
        corr_xyz0in1 = transform_points_Rt(corr_xyz0, Rt_gt)
        uv_0in1 = project_3dto2d(corr_xyz0in1, K_mat.clone())
        uv_1in1 = project_3dto2d(corr_xyz1, K_mat.clone())
        uv_0in0 = project_3dto2d(corr_xyz0, K_mat.clone())

        if 'no_filter' in filter_level_list:
            
            if i % dataset.num_tgt_per_src==0:
                if i != 0:
                    assert len(filtered_corr['no_filter'])==dataset.num_tgt_per_src
                   
                    np.savez_compressed(os.path.join(f"{dataset.root}/correspondence_dino",'no_filter',f"src_{src_name}_{save_name}.npz") ,name=filtered_corr['no_filter'])

                    filtered_corr = {key: [] for key in filter_level_list}
            filtered_corr['no_filter'].append({src_name: uv_0in0.numpy().astype(int), tgt_name: uv_1in1.numpy().astype(int)})
        
        if visualize:
            im0_show= instance["rgb_0o"]
            im1_show = instance["rgb_1o"]#.numpy().transpose((1,2,0))
            draw_corr(im1_show, im0_show , uv_1in1.numpy().astype(int), uv_0in0.numpy().astype(int), output_name=f'outputs/tgt_{tgt_name}_{src_name}')#f'outputs/src_{src_name}_{tgt_name}')
        
        corr_err2d = (uv_0in1 - uv_1in1).norm(p=2, dim=1)
        err_2d.append(corr_err2d.detach().cpu())
    import pdb 
    pdb.set_trace()
    err_2d = torch.stack(err_2d, dim=0).float()
    R_gt = torch.stack(R_gt, dim=0).float()

    """
    feats_0 = []
    feats_1 = []
    depth_0 = []
    depth_1 = []
    K_mat = []
    Rt_gt = []

    for batch in tqdm(loader):
        feat_0 = model(batch["rgb_0"].cuda())
        feat_1 = model(batch["rgb_1"].cuda())
        if cfg.multilayer:
            feat_0 = torch.cat(feat_0, dim=1)
            feat_1 = torch.cat(feat_1, dim=1)
        feats_0.append(feat_0.detach().cpu())
        feats_1.append(feat_1.detach().cpu())
        depth_0.append(batch["depth_0"])
        depth_1.append(batch["depth_1"])
        K_mat.append(batch["K"])
        Rt_gt.append(batch["Rt_1"])

    feats_0 = torch.cat(feats_0, dim=0)
    feats_1 = torch.cat(feats_1, dim=0)
    depth_0 = torch.cat(depth_0, dim=0)
    depth_1 = torch.cat(depth_1, dim=0)
    K_mat = torch.cat(K_mat, dim=0)
    Rt_gt = torch.cat(Rt_gt, dim=0).float()[:, :3, :4]

    depth_0 = nn_F.interpolate(depth_0, scale_factor=cfg.scale_factor)
    depth_1 = nn_F.interpolate(depth_1, scale_factor=cfg.scale_factor)
    K_mat[:, :2, :] *= cfg.scale_factor

    err_2d = []
    num_instances = len(loader.dataset)
    for i in tqdm(range(num_instances)):
        corr_xyz0, corr_xyz1, corr_dist = estimate_correspondence_depth(
            feats_0[i],
            feats_1[i],
            depth_0[i],
            depth_1[i],
            K_mat[i].clone(),
            cfg.num_corr,
        )

        corr_xyz0in1 = transform_points_Rt(corr_xyz0, Rt_gt[i].float())
        uv_0in1 = project_3dto2d(corr_xyz0in1, K_mat[i].clone())
        uv_1in1 = project_3dto2d(corr_xyz1, K_mat[i].clone())
        corr_err2d = (uv_0in1 - uv_1in1).norm(p=2, dim=1)

        err_2d.append(corr_err2d.detach().cpu())

    err_2d = torch.stack(err_2d, dim=0).float()
    """

    results = []
    # compute 2D errors
    px_thresh = [5, 10, 20]
    for _th in px_thresh:
        recall_i = 100 * (err_2d < _th).float().mean()
        print(f"Recall at {_th:>2d} pixels:  {recall_i:.2f}")
        results.append(f"{recall_i:5.02f}")

    # compute rel_ang
    rel_ang = so3_rotation_angle(R_gt)
    rel_ang = rel_ang * 180.0 / np.pi

    # compute thresholded recall
    rec_10px = (err_2d < 10).float().mean(dim=1)
    bin_rec = compute_binned_performance(rec_10px, rel_ang, [0, 15, 30, 60, 180])
    for bin_acc in bin_rec:
        results.append(f"{bin_acc * 100:5.02f}")

    # # result summary
    time = datetime.now().strftime("%d%m%Y-%H%M")
    exp_info = ", ".join(
        [
            f"{model.checkpoint_name:30s}",
            f"{model.patch_size:2d}",
            f"{str(model.layer):5s}",
            f"{model.output:10s}",
            loader.dataset.name,
            str(cfg.num_corr),
            str(cfg.scale_factor),
        ]
    )
    results = ", ".join(results)
    log = f"{time}, {exp_info}, {results} \n"
    with open("object_correspondence.log", "a") as f:
        f.write(log)


if __name__ == "__main__":
    main()
