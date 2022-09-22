import os
import pickle

import cv2
import numpy as np
import torch
import xlrd
# from termcolor import cprint

# from utils.mano_wrist import hands_mean, rot_pose_beta_to_mesh


COLOR_JOINTS = [
        [1.0, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, 0.6, 0.0],
        [0.0, 0.8, 0.0],
        [0.0, 1.0, 0.0],  # thumb
        [0.0, 0.0, 0.6],
        [0.0, 0.0, 1.0],
        [0.2, 0.2, 1.0],
        [0.4, 0.4, 1.0],  # index
        [0.0, 0.4, 0.4],
        [0.0, 0.6, 0.6],
        [0.0, 0.8, 0.8],
        [0.0, 1.0, 1.0],  # middle
        [0.4, 0.4, 0.0],
        [0.6, 0.6, 0.0],
        [0.8, 0.8, 0.0],
        [1.0, 1.0, 0.0],  # ring
        [0.4, 0.0, 0.4],
        [0.6, 0.0, 0.6],
        [0.8, 0.0, 0.8],
        [1.0, 0.0, 1.0],
    ]  # little


def plot_hand(image, coords_hw, vis=None, linewidth=3):
    """Plots a hand stick figure into a matplotlib figure."""

    colors = np.array(COLOR_JOINTS)
    colors = colors[:, ::-1]

    # define connections and colors of the bones
    bones = [
        ((0, 1), colors[1, :]),
        ((1, 2), colors[2, :]),
        ((2, 3), colors[3, :]),
        ((3, 4), colors[4, :]),
        ((0, 5), colors[5, :]),
        ((5, 6), colors[6, :]),
        ((6, 7), colors[7, :]),
        ((7, 8), colors[8, :]),
        ((0, 9), colors[9, :]),
        ((9, 10), colors[10, :]),
        ((10, 11), colors[11, :]),
        ((11, 12), colors[12, :]),
        ((0, 13), colors[13, :]),
        ((13, 14), colors[14, :]),
        ((14, 15), colors[15, :]),
        ((15, 16), colors[16, :]),
        ((0, 17), colors[17, :]),
        ((17, 18), colors[18, :]),
        ((18, 19), colors[19, :]),
        ((19, 20), colors[20, :]),
    ]

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        c1x = int(coord1[0])
        c1y = int(coord1[1])
        c2x = int(coord2[0])
        c2y = int(coord2[1])
        cv2.line(image, (c1x, c1y), (c2x, c2y), color=color * 255.0, thickness=linewidth)

    for i in range(coords_hw.shape[0]):
        cx = int(coords_hw[i, 0])
        cy = int(coords_hw[i, 1])
        cv2.circle(image, (cx, cy), radius=2 * linewidth, thickness=-1, color=colors[i, :] * 255.0)


# def test():
#     aa = pickle.load(open("20220508/2022-05-08_08-56-25/test.pkl", "rb"))

#     all_data = rot_pose_beta_to_mesh(aa["root"].view(1, 3), aa["pose"].contiguous().view(1, 45))
#     ref_joints = all_data[0, :21][[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
#     all_verts = all_data[0, 21:]

#     img = cv2.imread("20220508/2022-05-08_08-56-25/3.jpg")
#     h = img.shape[0]
#     w = img.shape[1]

#     scale_x = 100 * 30
#     scale_y = 100 * 27
#     ref_joints[:, 0] = ref_joints[:, 0] * scale_x
#     ref_joints[:, 1] = ref_joints[:, 1] * scale_y
#     u_shift = 0
#     v_shift = 0
#     # ref_joints[:, :2] = ref_joints[:, :2] / ref_joints[:, 2:]

#     ref_joints[:, 0] += u_shift
#     ref_joints[:, 1] += v_shift

#     # scale_x, scale_y
#     ref_joints[:, 0] = ref_joints[:, 0] - 30
#     ref_joints[:, 1] = ref_joints[:, 1] + 72

#     # import pdb; pdb.set_trace()

#     ref_joints[:, 0] = ref_joints[:, 0] + w // 2
#     ref_joints[:, 1] = h // 2 - ref_joints[:, 1] * 2
#     ref_joints = ref_joints.detach().cpu().numpy()

#     plot_hand(img, ref_joints)

#     cv2.circle(img, (w // 2, h // 2), radius=6, thickness=-1, color=(0, 0, 255))

#     cv2.imwrite("tmp.jpg", img)


def generate_2d(path, plot=False):
    img = cv2.imread(path + "/0.png")
    cv2.imwrite("tmp.jpg", img)

    excel_path = path + "/ExcelData.xls"
    screen_pos = []
    with xlrd.open_workbook(excel_path) as wb:
        sheet = wb.sheet_by_index(0)
        idx = 0
        for row in sheet.get_rows():
            if 1 <= idx <= 26:
                screen_pos.append(
                    torch.tensor([float(x) for x in row[8].value.split(",")], dtype=torch.float))
            idx += 1
    screen_pos_tensor = torch.stack(screen_pos)
    print(screen_pos_tensor.shape)
    ref_joints = screen_pos_tensor[[0, 22, 23, 24, 25, 17, 18, 19, 20, 12, 13, 14, 15, 7, 8, 9, 10, 2, 3, 4, 5]]

    ref_joints[:, 1] = 1 - ref_joints[:, 1]
    ref_joints[:, 0] *= img.shape[1]
    ref_joints[:, 1] *= img.shape[0]

    if plot:
        print(ref_joints)
        plot_hand(img, ref_joints)
        cv2.imwrite("tmp.jpg", img)
    return ref_joints


if __name__ == '__main__':
    data_dir = r"C:\Users\FengWang\Downloads\newhand\Hand\Build_Hand\ExportData\2022-08-02_17-07-21"
    generate_2d(data_dir, True)
