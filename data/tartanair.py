import os
import warnings
import random
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=UserWarning)


def generate_feature_map_for_ga(
    feature_fp,
    original_height=480,
    original_width=640,
    new_height=336,
    new_width=448,
    inverse_depth=False
):
    """
    Convert sparse point CSV/TXT into a dense sparse depth map of given size.
    CSV must have header 'row,column,depth'. TXT should be space-delimited.
    """
    ext = os.path.splitext(feature_fp)[-1].lower()
    if ext == '.csv':
        df = pd.read_csv(feature_fp)
    elif ext == '.txt':
        df = pd.read_csv(
            feature_fp,
            delimiter=' ',
            header=0,
            names=['row', 'column', 'depth']
        )
    else:
        raise ValueError("Unsupported format. Only CSV or TXT allowed.")

    sparse_depth = np.zeros((new_height, new_width), dtype=np.float32)
    scale_y = new_height / original_height
    scale_x = new_width / original_width

    valid_count = 0
    for _, r in df.iterrows():
        y = int(r['row'] * scale_y)
        x = int(r['column'] * scale_x)
        d = float(r['depth'])
        if inverse_depth:
            d = 1.0 / d
        if 0 <= y < new_height and 0 <= x < new_width:
            sparse_depth[y, x] = d
            valid_count += 1

    # return both map and count of valid points
    return sparse_depth[..., np.newaxis], valid_count


class TartanAir(Dataset):
    """
    Modified NYU dataset: reads a split .txt listing RGB, GT depth, and sparse CSV/TXT paths.
    Generates sparse depth via provided CSV/TXT instead of sampling.
    Output matches original NYU class: keys 'rgb','dep','gt','K','rgb_480640','dep_480640','num_sample'.
    """
    def __init__(self, args, mode):
        super().__init__()
        assert mode in ['train', 'val', 'test'], f"Invalid mode: {mode}"
        self.args = args
        self.mode = mode
        self.height = 336
        self.width = 448
        self.sparse_point_orig_h = 480
        self.sparse_point_orig_w = 640

        # camera intrinsics (half-resolution)
        self.K = torch.Tensor([
            1296.666758476217,
            1300.831316354508,
            501.50386149846,
            276.161712082695
        ])
        
        # read split txt
        self.sample_list = []
        path_txt = args.val_path_txt
        if mode == 'train':
            path_txt = args.train_path_txt
        with open(path_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rgb_fp, gt_fp, sp_fp = line.split()
                self.sample_list.append({
                    'rgb': rgb_fp,
                    'gt': gt_fp,
                    'sparse': sp_fp
                })

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        entry = self.sample_list[idx]
        rgb = Image.open(entry['rgb']).convert('RGB')
        dep_img = Image.open(entry['gt'])  # single-channel depth TIFF

        # augmentation
        
        t_rgb = T.Compose([
            T.Resize((self.height, self.width)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        t_dep = T.Compose([
            T.Resize((self.height, self.width)),
            T.ToTensor()
        ])

        rgb = t_rgb(rgb)
        dep = t_dep(dep_img)
        K = self.K.clone()

        # generate sparse depth map and count valid points
        sparse_np, ns = generate_feature_map_for_ga(
            entry['sparse'],
            original_height=self.sparse_point_orig_h,
            original_width=self.sparse_point_orig_w,
            new_height=self.height,
            new_width=self.width,
            inverse_depth=False
        )
        dep_sp = torch.from_numpy(sparse_np).permute(2, 0, 1)

        return {
            'rgb': rgb,
            'dep': dep_sp,
            'gt': dep,
            'K': K,
            'num_sample': ns
        }
