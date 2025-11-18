# ibrnet/data_loaders/dataset.py

import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class TransformsDataset(Dataset):
    """
    IBRNet train.py에서 사용하는 형태에 맞춘 실내(classroom) 데이터셋.

    __init__(self, args, mode, scenes=None)
    - args.rootdir : C:/NovelView_IBRNet/data/classrooms  같은 루트 폴더
    - scenes       : '609' 또는 ['609'] 같은 씬 이름(들)

    지금은 한 개 씬만 쓴다고 가정하고, 첫 번째 씬만 사용.
    """

    def __init__(self, args, mode, scenes=None):
        super().__init__()
        self.args = args
        self.mode = mode  # 'train' 또는 'val' 등이 들어올 수 있음

        # ----- rootdir / scene 이름 정리 -----
        rootdir = Path(args.rootdir)  # config 에서 rootdir로 설정할 예정

        if scenes is None:
            # rootdir 안의 모든 폴더를 씬으로 사용 (여러 방일 때)
            scene_names = [d.name for d in rootdir.iterdir() if d.is_dir()]
        elif isinstance(scenes, str):
            # '609' 처럼 문자열 하나 들어온 경우
            scene_names = [scenes]
        else:
            # 리스트 등
            scene_names = list(scenes)

        # 우선은 첫 번째 씬만 사용 (필요하면 나중에 여러 씬 지원으로 확장)
        self.scene_name = scene_names[0]
        self.scene_root = rootdir / self.scene_name

        # ----- transforms.json 로딩 -----
        json_path = self.scene_root / "transforms.json"
        if not json_path.exists():
            raise FileNotFoundError(f"transforms.json을 찾을 수 없습니다: {json_path}")

        with open(json_path, "r") as f:
            meta = json.load(f)

        self.W = int(meta["w"])
        self.H = int(meta["h"])

        frames = meta["frames"]

        self.image_paths = []
        self.c2w_list = []

        for fr in frames:
            rel = fr["file_path"]  # 예: "images_2x/img0001.jpg" 또는 "images/img0001.jpg"
            img_path = self.scene_root / rel
            if not img_path.exists():
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")
            self.image_paths.append(img_path)
            self.c2w_list.append(np.array(fr["transform_matrix"], dtype=np.float32))

        self.c2w = np.stack(self.c2w_list, axis=0)  # (N, 4, 4)
        self.n_images = len(self.image_paths)

        # ----- 이미지 미리 메모리에 올리기 -----
        imgs = []
        for p in self.image_paths:
            im = Image.open(p).convert("RGB")
            im = np.array(im, dtype=np.float32) / 255.0  # (H, W, 3)
            imgs.append(im)
        self.images = np.stack(imgs, axis=0)  # (N, H, W, 3)

        # ----- intrinsics (대충 pinhole) -----
        fx = fy = 0.5 * self.W
        cx = self.W * 0.5
        cy = self.H * 0.5
        self.intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        """
        train.py에서 요구하는 포맷에 맞춤

        반환:
            - target_rgb : (H, W, 3) float32 tensor
            - target_c2w : (4, 4) tensor
            - src_rgbs   : (N_src, H, W, 3) tensor
            - src_c2w    : (N_src, 4, 4) tensor
            - intrinsics : (4,) tensor [fx, fy, cx, cy]
            - scene_name : str
            - img_index  : int

        + IBRNet 호환용 추가 키:
            - rgb         : target_rgb와 동일
            - rgb_path    : 현재 타겟 이미지의 경로(str)
            - camera      : dict {H, W, intrinsics, c2w}
            - src_cameras : list of dicts
        """
        # ----- 타겟 뷰 -----
        tgt_rgb_np = self.images[idx]   # (H, W, 3) numpy
        tgt_c2w_np = self.c2w[idx]      # (4, 4)   numpy

        # 소스 인덱스: 나머지 모든 뷰
        all_indices = list(range(self.n_images))
        src_indices = [i for i in all_indices if i != idx]

        src_rgbs_np = self.images[src_indices]   # (N_src, H, W, 3)
        src_c2w_np  = self.c2w[src_indices]      # (N_src, 4, 4)

        # numpy → torch
        tgt_rgb = torch.from_numpy(tgt_rgb_np)       # (H, W, 3)
        tgt_c2w = torch.from_numpy(tgt_c2w_np)       # (4, 4)
        src_rgbs = torch.from_numpy(src_rgbs_np)     # (N_src, H, W, 3)
        src_c2w  = torch.from_numpy(src_c2w_np)      # (N_src, 4, 4)
        intr     = torch.from_numpy(self.intrinsics) # (4,)

        H, W = tgt_rgb.shape[:2]

        # ----- 기본 키들 -----
        data = {
            "target_rgb": tgt_rgb,
            "target_c2w": tgt_c2w,
            "src_rgbs": src_rgbs,
            "src_c2w": src_c2w,
            "intrinsics": intr,
            "scene_name": self.scene_name,
            "img_index": idx,
        }

        # ----- IBRNet 호환 키들 추가 -----
        # rgb: 타겟 이미지
        data["rgb"] = tgt_rgb

        # rgb_path: 현재 타겟 이미지의 경로 (문자열)
        # self.image_paths는 __init__에서 만듬
        data["rgb_path"] = str(self.image_paths[idx])

        # camera: 단일 타겟 뷰 카메라 정보
        data["camera"] = {
            "H": H,
            "W": W,
            "intrinsics": intr,   # [fx, fy, cx, cy] tensor
            "c2w": tgt_c2w,       # (4, 4) tensor
        }

        # src_cameras: 소스 뷰 카메라 정보 리스트
        src_cameras = []
        for i in range(src_c2w.shape[0]):
            src_cameras.append({
                "H": H,
                "W": W,
                "intrinsics": intr,
                "c2w": src_c2w[i],   # (4, 4) tensor
            })
        data["src_cameras"] = src_cameras
        # 실내 기준 drpth_range 설정
        data["depth_range"] = torch.tensor([0.1, 5.0], dtype=torch.float32)

        return data
