# coding: UTF-8

import os
import torch
import pickle
import numpy as np
from torch.autograd import Variable
from pytorch3d.io import load_obj, save_obj
from reprojection import generate_2d

    
obj_filename = os.path.join('extra_data/hand_wrist/hand_01.obj')
# obj_filename = os.path.join('extra_data/hand_mesh/hand.obj')
verts, faces, aux = load_obj(
            obj_filename,
            device="cuda",
            load_textures=True,
            create_texture_atlas=True,
            texture_atlas_size=8,
            texture_wrap=None,
        )
atlas = aux.texture_atlas
mesh_num = 842
keypoints_num = 16

dd = pickle.load(open('extra_data/MANO_RIGHT.pkl', 'rb'),encoding='latin1')
kintree_table = dd['kintree_table']
id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
parent = {i: id_to_col[kintree_table[0, i]] for i in range(1, kintree_table.shape[1])} 

mesh_mu = verts.unsqueeze(0)  # [bs, 778, 3] ---> [bs, 842, 3](hand_01/02)
posedirs = Variable(torch.from_numpy(np.expand_dims(dd['posedirs'], 0).astype(np.float32)).cuda())   # [bs, 778, 3, 135] (rot_pose_beta_mesh)
J_regressor = Variable(torch.from_numpy(np.expand_dims(dd['J_regressor'].todense(), 0).astype(np.float32)).cuda()) # [bs, 16, 778] (rot_pose_beta_mesh)
weights = Variable(torch.from_numpy(np.expand_dims(dd['weights'], 0).astype(np.float32)).cuda())     # [bs, 778, 16] (rot_pose_beta_mesh)
# hands_components = Variable(
#     torch.from_numpy(np.expand_dims(np.vstack(dd['hands_components'][:]), 0).astype(np.float32)).cuda())
hands_mean = Variable(torch.from_numpy(np.expand_dims(dd['hands_mean'], 0).astype(np.float32)).cuda()) # [bs, 45] unchanged
root_rot = Variable(torch.FloatTensor([0., 0., 0.]).unsqueeze(0).cuda())
FACES = faces[0] 

# set extra vertex fpr J_regressor & posedirs & weights

Final_J_regressor = torch.zeros(1, 16, 842)
Final_J_regressor[:, :, :778] = J_regressor

Final_J_regressor[:, :, 793] = J_regressor[:, :, 777]
Final_J_regressor[:, :, 809] = J_regressor[:, :, 777]
Final_J_regressor[:, :, 825] = J_regressor[:, :, 777]
Final_J_regressor[:, :, 841] = J_regressor[:, :, 777]

# Final_J_regressor[:, :, 788] = J_regressor[:, :, 235]
# Final_J_regressor[:, :, 804] = J_regressor[:, :, 235]
# Final_J_regressor[:, :, 820] = J_regressor[:, :, 235]
# Final_J_regressor[:, :, 836] = J_regressor[:, :, 235]

# Final_J_regressor[:, :, 787] = J_regressor[:, :, 230]
# Final_J_regressor[:, :, 803] = J_regressor[:, :, 230]
# Final_J_regressor[:, :, 819] = J_regressor[:, :, 230]
# Final_J_regressor[:, :, 835] = J_regressor[:, :, 230]

# Final_J_regressor[:, :, 781] = J_regressor[:, :, 92]
# Final_J_regressor[:, :, 797] = J_regressor[:, :, 92]
# Final_J_regressor[:, :, 813] = J_regressor[:, :, 92]
# Final_J_regressor[:, :, 829] = J_regressor[:, :, 92]

# Final_J_regressor[:, :, 780] = J_regressor[:, :, 39]
# Final_J_regressor[:, :, 796] = J_regressor[:, :, 39]
# Final_J_regressor[:, :, 812] = J_regressor[:, :, 39]
# Final_J_regressor[:, :, 828] = J_regressor[:, :, 39]

# Final_J_regressor[:, :, 789] = J_regressor[:, :, 288]
# Final_J_regressor[:, :, 805] = J_regressor[:, :, 288]
# Final_J_regressor[:, :, 821] = J_regressor[:, :, 288]
# Final_J_regressor[:, :, 837] = J_regressor[:, :, 288]

# Final_J_regressor[:, :, 784] = J_regressor[:, :, 118]
# Final_J_regressor[:, :, 800] = J_regressor[:, :, 118]
# Final_J_regressor[:, :, 815] = J_regressor[:, :, 118]
# Final_J_regressor[:, :, 831] = J_regressor[:, :, 118]

# Final_J_regressor[:, :, 783] = J_regressor[:, :, 117]
# Final_J_regressor[:, :, 799] = J_regressor[:, :, 117]
# Final_J_regressor[:, :, 816] = J_regressor[:, :, 117]
# Final_J_regressor[:, :, 832] = J_regressor[:, :, 117]

# Final_J_regressor[:, :, 785] = J_regressor[:, :, 119]
# Final_J_regressor[:, :, 801] = J_regressor[:, :, 119]
# Final_J_regressor[:, :, 817] = J_regressor[:, :, 119]
# Final_J_regressor[:, :, 833] = J_regressor[:, :, 119]

# Final_J_regressor[:, :, 786] = J_regressor[:, :, 120]
# Final_J_regressor[:, :, 802] = J_regressor[:, :, 120]
# Final_J_regressor[:, :, 818] = J_regressor[:, :, 120]
# Final_J_regressor[:, :, 834] = J_regressor[:, :, 120]

# Final_J_regressor[:, :, 782] = J_regressor[:, :, 108]
# Final_J_regressor[:, :, 798] = J_regressor[:, :, 108]
# Final_J_regressor[:, :, 814] = J_regressor[:, :, 108]
# Final_J_regressor[:, :, 830] = J_regressor[:, :, 108]

# Final_J_regressor[:, :, 778] = J_regressor[:, :, 79]
# Final_J_regressor[:, :, 795] = J_regressor[:, :, 79]
# Final_J_regressor[:, :, 811] = J_regressor[:, :, 79]
# Final_J_regressor[:, :, 827] = J_regressor[:, :, 79]

# Final_J_regressor[:, :, 779] = J_regressor[:, :, 78]
# Final_J_regressor[:, :, 794] = J_regressor[:, :, 78]
# Final_J_regressor[:, :, 810] = J_regressor[:, :, 78]
# Final_J_regressor[:, :, 826] = J_regressor[:, :, 78]

# Final_J_regressor[:, :, 790] = J_regressor[:, :, 774]
# Final_J_regressor[:, :, 806] = J_regressor[:, :, 774]
# Final_J_regressor[:, :, 822] = J_regressor[:, :, 774]
# Final_J_regressor[:, :, 838] = J_regressor[:, :, 774]

# Final_J_regressor[:, :, 791] = J_regressor[:, :, 775]
# Final_J_regressor[:, :, 807] = J_regressor[:, :, 775]
# Final_J_regressor[:, :, 823] = J_regressor[:, :, 775]
# Final_J_regressor[:, :, 839] = J_regressor[:, :, 775]

# Final_J_regressor[:, :, 792] = J_regressor[:, :, 776]
# Final_J_regressor[:, :, 808] = J_regressor[:, :, 776]
# Final_J_regressor[:, :, 824] = J_regressor[:, :, 776]
# Final_J_regressor[:, :, 840] = J_regressor[:, :, 776]


Final_posedirs = torch.zeros(1, 842, 3, 135)
Final_posedirs[:, :778, :, :] = posedirs

Final_weights = torch.zeros(1, 842, 16)
Final_weights[:, :778] = weights
Final_weights[:, 778:] = weights[:, 777]

Final_J_regressor = Final_J_regressor.cuda()
Final_weights = Final_weights.cuda()
Final_posedirs = Final_posedirs.cuda()


def rodrigues(r):
    #print(r)
    theta = torch.sqrt(torch.sum(torch.pow(r, 2), 1))

    def S(n_):
        ns = torch.split(n_, 1, 1)
        Sn_ = torch.cat([torch.zeros_like(ns[0]), -ns[2], ns[1], ns[2], torch.zeros_like(ns[0]), -ns[0], -ns[1], ns[0],
                         torch.zeros_like(ns[0])], 1)
        Sn_ = Sn_.view(-1, 3, 3)
        return Sn_

    n = r / (theta.view(-1, 1))
    Sn = S(n)

    # R = torch.eye(3).unsqueeze(0) + torch.sin(theta).view(-1, 1, 1)*Sn\
    #        +(1.-torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn,Sn)

    I3 = Variable(torch.eye(3).unsqueeze(0).cuda())
    #print(theta,Sn)
    R = I3 + torch.sin(theta).view(-1, 1, 1) * Sn \
        + (1. - torch.cos(theta).view(-1, 1, 1)) * torch.matmul(Sn, Sn)

    Sr = S(r)
    theta2 = theta ** 2
    R2 = I3 + (1. - theta2.view(-1, 1, 1) / 6.) * Sr \
         + (.5 - theta2.view(-1, 1, 1) / 24.) * torch.matmul(Sr, Sr)

    idx = np.argwhere((theta < 1e-30).data.cpu().numpy())

    if (idx.size):
        R[idx, :, :] = R2[idx, :, :]

    return R, Sn


def get_poseweights(poses, bsize):
    # pose: batch x 24 x 3
    pose_matrix, _ = rodrigues(poses[:, 1:, :].contiguous().view(-1, 3))
    # pose_matrix, _ = rodrigues(poses.view(-1,3))
    pose_matrix = pose_matrix - Variable(torch.from_numpy(
        np.repeat(np.expand_dims(np.eye(3, dtype=np.float32), 0), bsize * (keypoints_num - 1), axis=0)).cuda())
    pose_matrix = pose_matrix.view(bsize, -1)
    return pose_matrix

        
# NOTICE: remove shape parameter.
def rot_pose_beta_to_mesh(rots, poses):
    
    batch_size = rots.size(0)
    #print(hands_mean.shape,poses.unsqueeze(1).shape,hands_components.shape)
    #poses = (hands_mean + torch.matmul(poses.unsqueeze(1), hands_components).squeeze(1)).view(batch_size,keypoints_num - 1, 3)  #if use pca
    poses = (hands_mean + poses).view(batch_size, keypoints_num - 1, 3)
    # poses = torch.cat((poses[:,:3].contiguous().view(batch_size,1,3),poses_),1)
    poses = torch.cat((root_rot.repeat(batch_size, 1).view(batch_size, 1, 3), poses), 1)

    v_shaped = mesh_mu.repeat(batch_size, 1, 1).view(batch_size, -1).view(batch_size, mesh_num, 3)
    pose_weights = get_poseweights(poses, batch_size)
    
    v_posed = v_shaped + torch.matmul(Final_posedirs.repeat(batch_size, 1, 1, 1),
                                      (pose_weights.view(batch_size, 1, (keypoints_num - 1) * 9, 1)).repeat(1, mesh_num,
                                                                                                            1,
                                                                                                            1)).squeeze(
        3)

    J_posed = torch.matmul(v_shaped.permute(0, 2, 1), Final_J_regressor.repeat(batch_size, 1, 1).permute(0, 2, 1))
    J_posed = J_posed.permute(0, 2, 1)
    J_posed_split = [sp.contiguous().view(batch_size, 3) for sp in torch.split(J_posed.permute(1, 0, 2), 1, 0)]

    pose = poses.permute(1, 0, 2)
    pose_split = torch.split(pose, 1, 0)
    

    angle_matrix = []
    for i in range(keypoints_num):
        #print(i, pose_split[i])
        out, tmp = rodrigues(pose_split[i].contiguous().view(-1, 3))
        angle_matrix.append(out)

    # with_zeros = lambda x: torch.cat((x,torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size,1,1)),1)

    with_zeros = lambda x: \
        torch.cat((x, Variable(torch.FloatTensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(batch_size, 1, 1).cuda())), 1)

    pack = lambda x: torch.cat((Variable(torch.zeros(batch_size, 4, 3).cuda()), x), 2)

    results = {}
    results[0] = with_zeros(torch.cat((angle_matrix[0], J_posed_split[0].view(batch_size, 3, 1)), 2))

    for i in range(1, kintree_table.shape[1]):
        tmp = with_zeros(torch.cat((angle_matrix[i],
                                    (J_posed_split[i] - J_posed_split[parent[i]]).view(batch_size, 3, 1)), 2))
        results[i] = torch.matmul(results[parent[i]], tmp)

    results_global = results

    results2 = []

    for i in range(len(results)):
        vec = (torch.cat((J_posed_split[i], Variable(torch.zeros(batch_size, 1).cuda())), 1)).view(batch_size, 4, 1)
        results2.append((results[i] - pack(torch.matmul(results[i], vec))).unsqueeze(0))

    results = torch.cat(results2, 0)

    T = torch.matmul(results.permute(1, 2, 3, 0),
                     Final_weights.repeat(batch_size, 1, 1).permute(0, 2, 1).unsqueeze(1).repeat(1, 4, 1, 1))
    Ts = torch.split(T, 1, 2)
    rest_shape_h = torch.cat((v_posed, Variable(torch.ones(batch_size, mesh_num, 1).cuda())), 2)
    rest_shape_hs = torch.split(rest_shape_h, 1, 2)

    v = Ts[0].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[0].contiguous().view(-1, 1, mesh_num) \
        + Ts[1].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[1].contiguous().view(-1, 1, mesh_num) \
        + Ts[2].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[2].contiguous().view(-1, 1, mesh_num) \
        + Ts[3].contiguous().view(batch_size, 4, mesh_num) * rest_shape_hs[3].contiguous().view(-1, 1, mesh_num)

    # v = v.permute(0,2,1)[:,:,:3]
    Rots = rodrigues(rots)[0]

    Jtr = []

    for j_id in range(len(results_global)):
        Jtr.append(results_global[j_id][:, :3, 3:4])
    
    #definition as frankmocap smplx @meshlab
    # Jtr.append(v[:, :3, 333].unsqueeze(2)) #index
    # Jtr.append(v[:, :3, 444].unsqueeze(2)) #middle
    # Jtr.append(v[:, :3, 672].unsqueeze(2)) #pinky
    # Jtr.append(v[:, :3, 555].unsqueeze(2)) #ring
    # Jtr.append(v[:, :3, 745].unsqueeze(2)) #thumb
    
    # 2022.05.11 lixin.yang
    Jtr.append(v[:, :3, 745].unsqueeze(2)) #thumb
    Jtr.append(v[:, :3, 317].unsqueeze(2)) #index
    Jtr.append(v[:, :3, 444].unsqueeze(2)) #middle
    Jtr.append(v[:, :3, 556].unsqueeze(2)) #ring
    Jtr.append(v[:, :3, 673].unsqueeze(2)) #pinky

    Jtr = torch.cat(Jtr, 2)  # .permute(0,2,1)

    v = torch.matmul(Rots, v[:, :3, :]).permute(0, 2, 1)  # .contiguous().view(batch_size,-1)
    Jtr = torch.matmul(Rots, Jtr).permute(0, 2, 1)  # .contiguous().view(batch_size,-1)
    
    #translate to be same as smplx
    root=Jtr[:,1].clone().unsqueeze(1)
    Jtr-=root
    v-=root
    
    return torch.cat((Jtr, v), 1)