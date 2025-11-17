# -*- coding: utf-8 -*-
"""
COLMAP outputs -> NeRF/IBRNet-friendly transforms.json + sparse depth
- Reads cameras.txt, images.txt, points3D.txt
- Saves per-image pose (w2c), intrinsics, and exports sparse_points.ply
"""
import os, json, math
import numpy as np

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

def parse_colmap_sparse(colmap_sparse_dir, images_dir):
    cams = {}
    with open(os.path.join(colmap_sparse_dir, 'cameras.txt'), 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            elems = line.strip().split()
            cam_id, model = int(elems[0]), elems[1]
            w, h = int(elems[2]), int(elems[3])
            params = list(map(float, elems[4:]))
            cams[cam_id] = (model, w, h, params)

    images = {}
    with open(os.path.join(colmap_sparse_dir, 'images.txt'), 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            elems = line.strip().split()
            if len(elems) < 10:  # image line
                continue
            img_id = int(elems[0])
            q = list(map(float, elems[1:5]))
            t = np.array(list(map(float, elems[5:8])))
            cam_id = int(elems[8])
            name = elems[9]
            R = qvec2rotmat(q)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3,:3] = R.T
            c2w[:3, 3] = (-R.T @ t.reshape(3,1)).flatten()
            images[name] = dict(c2w=c2w, cam_id=cam_id)

    # points3D (for DS-NeRF sparse depth)
    pts = []
    ply_path = os.path.join(colmap_sparse_dir, 'points3D.txt')
    if os.path.exists(ply_path):
        with open(ply_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                elems = line.strip().split()
                X = list(map(float, elems[1:4]))
                R,G,B = list(map(float, elems[4:7]))
                pts.append(X)
    sparse_points = np.array(pts, dtype=np.float32) if pts else None

    return cams, images, sparse_points

def to_transforms_json(cams, images, images_dir, out_json):
    frames=[]
    any_cam = list(cams.values())[0]
    _, W, H, params = any_cam
    fl = params[0] if len(params)>0 else max(W,H) * 1.2
    for name, rec in images.items():
        c2w = rec['c2w'].tolist()
        frames.append({"file_path": os.path.join("images", name), "transform_matrix": c2w})
    data = {
        "w": W, "h": H, "fl_x": fl, "fl_y": fl,
        "cx": W/2.0, "cy": H/2.0,
        "frames": frames
    }
    with open(out_json,'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    import argparse, pathlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--room_dir", required=True, help="e.g., /classrooms/roomA")
    args = ap.parse_args()
    sparse_dir = os.path.join(args.room_dir, "colmap", "sparse", "0")
    cams, images, sparse = parse_colmap_sparse(sparse_dir, os.path.join(args.room_dir,"images"))
    to_transforms_json(cams, images, os.path.join(args.room_dir,"images"),
                       os.path.join(args.room_dir, "transforms.json"))
    if sparse is not None:
        np.save(os.path.join(args.room_dir,"sparse_points.npy"), sparse)
    print("OK -> transforms.json, sparse_points.npy (optional)")
