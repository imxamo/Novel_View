# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn.functional as F


rng = np.random.RandomState(234)

########################################################################################################################
# ray batch sampling
########################################################################################################################


def parse_camera(params):
    if isinstance(params, dict):
        H_val = int(params["H"])
        W_val = int(params["W"])

        intr = params["intrinsics"]  # 가능성: (4,), (1,4), (3,3), (1,3,3)

        if isinstance(intr, np.ndarray):
            intr_t = torch.from_numpy(intr).float()
        elif isinstance(intr, torch.Tensor):
            intr_t = intr.float()
        else:
            intr_t = torch.tensor(intr, dtype=torch.float32)

        # 모양에 따라
        # (4,) 또는 (1,4) → [fx, fy, cx, cy]로 해석
        if (intr_t.ndim == 1 and intr_t.numel() == 4) or \
           (intr_t.ndim == 2 and intr_t.shape == (1, 4)):
            intr_flat = intr_t.view(-1)  # (4,)
            fx, fy, cx, cy = intr_flat
            K = torch.zeros(1, 3, 3, dtype=intr_t.dtype, device=intr_t.device)
            K[0, 0, 0] = fx
            K[0, 1, 1] = fy
            K[0, 0, 2] = cx
            K[0, 1, 2] = cy
            K[0, 2, 2] = 1.0

        elif intr_t.ndim == 2 and intr_t.shape == (3, 3):
            # 이미 3x3 K인 경우
            K = intr_t.unsqueeze(0)

        elif intr_t.ndim == 3 and intr_t.shape[1:] == (3, 3):
            # (1,3,3) 같은 경우
            K = intr_t

        else:
            raise ValueError(f"Unexpected intrinsics shape: {intr_t.shape}")

        c2w = params["c2w"]  # 가능성: (4,4) 또는 (1,4,4)
        if isinstance(c2w, np.ndarray):
            c2w_t = torch.from_numpy(c2w).float()
        elif isinstance(c2w, torch.Tensor):
            c2w_t = c2w.float()
        else:
            c2w_t = torch.tensor(c2w, dtype=torch.float32)

        if c2w_t.ndim == 2 and c2w_t.shape == (4, 4):
            c2w_t = c2w_t.unsqueeze(0)  # (1,4,4)
        elif c2w_t.ndim == 3 and c2w_t.shape[1:] == (4, 4):
            pass  # 이미 (B,4,4)
        else:
            raise ValueError(f"Unexpected c2w shape: {c2w_t.shape}")

        H_t = torch.tensor([H_val], dtype=torch.float32, device=K.device)  # shape: (1,)
        W_t = torch.tensor([W_val], dtype=torch.float32, device=K.device)  # shape: (1,)

        # IBRNet 쪽에서 기대하는 반환 형태
        # (W, H, intrinsics=(1,3,3), c2w=(1,4,4))
        return W_t, H_t, K, c2w_t

    H = params[:, 0]
    W = params[:, 1]
    fx = params[:, 2]
    fy = params[:, 3]
    cx = params[:, 4]
    cy = params[:, 5]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2w = params[:, 18:34].reshape((-1, 4, 4))
    
    return W, H, intrinsics, c2w

def dilate_img(img, kernel_size=20):
    import cv2
    assert img.dtype == np.uint8
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv2.dilate(img / 255, kernel, iterations=1) * 255
    return dilation


class RaySamplerSingleImage(object):
    def __init__(self, data, device, resize_factor=1, render_stride=1):
        super().__init__()
        self.render_stride = render_stride
        """
        # 키 통일 로직
        # DU/TransformsDataset에서 온 배치라면 'camera'가 없음
        if 'camera' not in data:
            # dataset.py에서 출력했던 키 이용해야 함
            # target_rgb: [H, W, 3]
            target_rgb = data['target_rgb']  # torch.Tensor
            H, W = target_rgb.shape[:2]

            intr = data['intrinsics']        # [fx, fy, cx, cy]
            c2w  = data['target_c2w']        # [4, 4]

            # sample_ray에서 실제로 어떤 필드를 쓰는지에 따라 더 추가할 수 있음
            # 우선 최소한 해상도 + intrinsics + pose를 넣어 준다.
            data['camera'] = {
                'H': H,
                'W': W,
                'intrinsics': intr,
                'c2w': c2w,
            }
        """
        
        self.rgb = data['rgb'] if 'rgb' in data.keys() else None
        self.camera = data['camera']
        self.rgb_path = data['rgb_path']
        self.depth_range = data['depth_range']
        self.device = device
        W, H, self.intrinsics, self.c2w_mat = parse_camera(self.camera)
        self.batch_size = len(self.camera)

        self.H = int(H[0])
        self.W = int(W[0])

        # half-resolution output
        if resize_factor != 1:
            self.W = int(self.W * resize_factor)
            self.H = int(self.H * resize_factor)
            self.intrinsics[:, :2, :3] *= resize_factor
            if self.rgb is not None:
                self.rgb = F.interpolate(self.rgb.permute(0, 3, 1, 2), scale_factor=resize_factor).permute(0, 2, 3, 1)

        self.rays_o, self.rays_d = self.get_rays_single_image(self.H, self.W, self.intrinsics, self.c2w_mat)
        if self.rgb is not None:
            self.rgb = self.rgb.reshape(-1, 3)

        if 'src_rgbs' in data.keys():
            self.src_rgbs = data['src_rgbs']
        else:
            self.src_rgbs = None
        if 'src_cameras' in data.keys():
            self.src_cameras = data['src_cameras']
        else:
            self.src_cameras = None

    def get_rays_single_image(self, H, W, intrinsics, c2w):
        '''
        :param H: image height
        :param W: image width
        :param intrinsics: 4 by 4 intrinsic matrix
        :param c2w: 4 by 4 camera to world extrinsic matrix
        :return:
        '''
        u, v = np.meshgrid(np.arange(W)[::self.render_stride], np.arange(H)[::self.render_stride])
        u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
        pixels = torch.from_numpy(pixels)
        batched_pixels = pixels.unsqueeze(0).repeat(self.batch_size, 1, 1)

        rays_d = (c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)).transpose(1, 2)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)  # B x HW x 3
        return rays_o, rays_d

    def get_all(self):
        ret = {'ray_o': self.rays_o.cuda(),
               'ray_d': self.rays_d.cuda(),
               'depth_range': self.depth_range.cuda(),
               'camera': self.camera.cuda(),
               'rgb': self.rgb.cuda() if self.rgb is not None else None,
               'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
               'src_cameras': self.src_cameras.cuda() if self.src_cameras is not None else None,
        }
        return ret

    def sample_random_pixel(self, N_rand, sample_mode, center_ratio=0.8):
        if sample_mode == 'center':
            border_H = int(self.H * (1 - center_ratio) / 2.)
            border_W = int(self.W * (1 - center_ratio) / 2.)

            # pixel coordinates
            u, v = np.meshgrid(np.arange(border_H, self.H - border_H),
                               np.arange(border_W, self.W - border_W))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = rng.choice(u.shape[0], size=(N_rand,), replace=False)
            select_inds = v[select_inds] + self.W * u[select_inds]

        elif sample_mode == 'uniform':
            # Random from one image
            select_inds = rng.choice(self.H*self.W, size=(N_rand,), replace=False)
        else:
            raise Exception("unknown sample mode!")

        return select_inds

    def random_sample(self, N_rand, sample_mode, center_ratio=0.8):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''

        select_inds = self.sample_random_pixel(N_rand, sample_mode, center_ratio)

        rays_o = self.rays_o[select_inds]
        rays_d = self.rays_d[select_inds]

        if self.rgb is not None:
            rgb = self.rgb[select_inds]
        else:
            rgb = None

        ret = {'ray_o': rays_o.cuda(),
               'ray_d': rays_d.cuda(),
               'camera': self.camera.cuda(),
               'depth_range': self.depth_range.cuda(),
               'rgb': rgb.cuda() if rgb is not None else None,
               'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
               'src_cameras': self.src_cameras.cuda() if self.src_cameras is not None else None,
               'selected_inds': select_inds
        }
        return ret
