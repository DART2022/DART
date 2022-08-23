import os
import pickle

import cv2
import imageio
import numpy as np
from manotorch.manolayer import ManoLayer
from pytorch3d.io import load_obj

from DARTset_utils import (aa_to_rotmat, fit_ortho_param, ortho_project, plot_hand, rotmat_to_aa)

RAW_IMAGE_SIZE = 512
BG_IMAGE_SIZE = 384
DATA_ROOT = "./data"


class DARTset():

    def __init__(self, data_split="train", use_full_wrist=True, load_wo_background=False):

        self.name = "DARTset"
        self.data_split = data_split
        self.root = os.path.join(DATA_ROOT, self.name, self.data_split)
        self.load_wo_background = load_wo_background
        self.raw_img_size = RAW_IMAGE_SIZE
        self.img_size = RAW_IMAGE_SIZE if load_wo_background else BG_IMAGE_SIZE

        self.use_full_wrist = use_full_wrist

        self.MANO_pose_mean = ManoLayer(joint_rot_mode="axisang",
                                        use_pca=False,
                                        mano_assets_root="assets/mano_v1_2",
                                        center_idx=0,
                                        flat_hand_mean=False).th_hands_mean.numpy().reshape(-1)

        obj_filename = os.path.join('./assets/hand_mesh.obj')
        _, faces, _ = load_obj(
            obj_filename,
            device="cpu",
            load_textures=False,
        )
        self.reorder_idx = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18]
        self.hand_faces = faces[0].numpy()

        self.load_dataset()

    def load_dataset(self):

        self.image_paths = []
        self.raw_mano_param = []
        self.joints_3d = []
        self.verts_3d_paths = []
        self.joints_2d = []

        image_parts = [
            r for r in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, r)) and "verts" not in r and "wbg" not in r
        ]
        image_parts = sorted(image_parts)

        for imgs_dir in image_parts:
            imgs_path = os.path.join(self.root, imgs_dir)
            data_record = pickle.load(open(os.path.join(self.root, f"part_{imgs_dir}.pkl"), "rb"))
            for k in range(len(data_record["pose"])):
                self.image_paths.append(os.path.join(imgs_path, data_record["img"][k]))
                self.raw_mano_param.append(data_record["pose"][k].astype(np.float32))
                self.joints_3d.append(data_record["joint3d"][k].astype(np.float32))
                self.joints_2d.append(data_record["joint2d"][k].astype(np.float32))
                verts_3d_path = os.path.join(imgs_path + "_verts", data_record["img"][k].replace(".png", ".pkl"))
                self.verts_3d_paths.append(verts_3d_path)

        self.sample_idxs = list(range(len(self.image_paths)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return {
            "image": self.get_image(idx),
            "joints_3d": self.get_joints_3d(idx),
            "joints_2d": self.get_joints_2d(idx),
            "joints_uvd": self.get_joints_uvd(idx),
            "verts_uvd": self.get_verts_uvd(idx),
            "ortho_intr": self.get_ortho_intr(idx),
            "sides": self.get_sides(idx),
            "image_mask": self.get_image_mask(idx),
        }

    def get_joints_3d(self, idx):
        joints = self.joints_3d[idx].copy()
        joints[:, 1:] = -joints[:, 1:]
        joints = joints[self.reorder_idx]
        joints = joints - joints[9] + np.array(
            [0, 0, 0.5])  # * We use ortho projection, so we need to shift the center of the hand to the origin
        return joints

    def get_verts_3d(self, idx):
        verts = pickle.load(open(self.verts_3d_paths[idx], "rb"))
        verts[:, 1:] = -verts[:, 1:]
        verts = verts + self.get_joints_3d(idx)[5]
        if not self.use_full_wrist:
            verts = verts[:778]
        verts = verts.astype(np.float32)
        return verts

    def get_joints_2d(self, idx):
        joints_2d = self.joints_2d[idx].copy()[self.reorder_idx]
        joints_2d = joints_2d / self.raw_img_size * self.img_size
        return joints_2d

    def get_image_path(self, idx):
        return self.image_paths[idx]

    def get_ortho_intr(self, idx):
        ortho_cam = fit_ortho_param(self.get_joints_3d(idx), self.get_joints_2d(idx))
        return ortho_cam

    def get_image(self, idx):
        path = self.image_paths[idx]
        if self.load_wo_background:
            img = np.array(imageio.imread(path, pilmode="RGBA"), dtype=np.uint8)
            img = img[:, :, :3]
        else:
            path = os.path.join(*path.split("/")[:-2], path.split("/")[-2] + "_wbg", path.split("/")[-1])
            img = cv2.imread(path)[..., ::-1]

        return img

    def get_image_mask(self, idx):
        path = self.image_paths[idx]
        image = np.array(imageio.imread(path, pilmode="RGBA"), dtype=np.uint8)
        image = cv2.resize(image, dsize=(self.img_size, self.img_size))
        return (image[:, :, 3] >= 128).astype(np.float32) * 255.0

    def get_joints_uvd(self, idx):
        uv = self.get_joints_2d(idx)
        d = self.get_joints_3d(idx)[:, 2:]  # (21, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_verts_uvd(self, idx):
        v3d = self.get_verts_3d(idx)
        ortho_cam = self.get_ortho_intr(idx)
        ortho_proj_verts = ortho_project(v3d, ortho_cam)
        d = v3d[:, 2:]
        uvd = np.concatenate((ortho_proj_verts, d), axis=1)
        return uvd

    def get_mano_pose(self, idx):
        pose = self.get_raw_mano_param(idx)  # [16, 3]
        unity2cam = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float32)
        root = rotmat_to_aa(unity2cam @ aa_to_rotmat(pose[0]))[None]
        new_pose = np.concatenate([root.reshape(-1), pose[1:].reshape(-1) + self.MANO_pose_mean], axis=0)  # [48]
        return new_pose.astype(np.float32)

    def get_mano_shape(self, idx):
        return np.zeros((10), dtype=np.float32)

    def get_sides(self, idx):
        return "right"


if __name__ == "__main__":

    dart_set = DARTset(data_split="test")

    for i in range(len(dart_set)):
        output = dart_set[i]

        image = output["image"]
        mask = (output["image_mask"]).astype(np.uint8)

        joints_2d = output["joints_2d"]
        joints_3d = output["joints_3d"]
        joints_uvd = output["joints_uvd"]
        verts_uvd = output["verts_uvd"]
        ortho_intr = output["ortho_intr"]

        proj_2d = ortho_project(joints_3d, ortho_intr)

        frame_1 = image.copy()
        mask = mask[:, :, None]
        mask = np.concatenate([mask, mask * 0, mask * 0], axis=2)
        frame_2 = cv2.addWeighted(frame_1, 0.5, mask, 0.5, 0)

        all_2d_opt = {"ortho_proj": proj_2d, "gt": joints_2d, "uv": joints_uvd[:, :2]}
        plot_hand(frame_1, all_2d_opt["uv"], linewidth=1)
        plot_hand(frame_1, all_2d_opt["gt"], linewidth=1)
        plot_hand(frame_1, all_2d_opt["ortho_proj"], linewidth=1)

        img_list = [image, frame_1, frame_2]
        comb_image = np.hstack(img_list)

        comb_image = cv2.cvtColor(comb_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("comb_image", comb_image)
        cv2.waitKey(0)
