import torch
from pytorch3d.transforms import axis_angle_to_quaternion,\
    quaternion_invert, quaternion_multiply, matrix_to_quaternion

MANO_INDEX = []
MANO_ROT_PARENT = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]


class FingerConfig:

    def __init__(self, name, pos_index, rot_index):
        self.name = name
        self.pos_index = pos_index
        self.rot_index = rot_index


INDEX = FingerConfig("index", [1, 2, 3, 16], [1, 2, 3])
MIDDLE = FingerConfig("middle", [4, 5, 6, 17], [4, 5, 6])
PINKY = FingerConfig("pinky", [7, 8, 9, 18], [7, 8, 9])
RING = FingerConfig("ring", [10, 11, 12, 19], [10, 11, 12])
THUMB = FingerConfig("thumb", [13, 14, 15, 20], [13, 14, 15])

WRIST_ROTATION = torch.tensor([ 0.7598, -0.6481,  0.0243,  0.0458], dtype=torch.float)


def norm(val: torch.Tensor):
    if val.norm() > 0:
        return val / val.norm()
    else:
        return val


def look_rotation(forward, up) -> torch.Tensor:
    forward = norm(forward)
    up = norm(up)
    right = norm(torch.cross(up, forward))
    rot_mat = torch.eye(3)
    for i in range(3):
        rot_mat[i] = torch.tensor([right[i], up[i], forward[i]])
    return norm(matrix_to_quaternion(rot_mat))


def convert_local_to_global(rotation):
    for i in range(rotation.shape[0]):
        if MANO_ROT_PARENT[i] >= 0:
            rotation[i] = norm(quaternion_multiply(rotation[MANO_ROT_PARENT[i]], rotation[i]))
    return rotation


def convert_global_to_local(rotation):
    for i in range(15, -1, -1):
        parent = MANO_ROT_PARENT[i]
        if parent >= 0:
            rotation[i] = norm(quaternion_multiply(quaternion_invert(rotation[parent]), rotation[i]))
    return rotation[1:]


def calc_hand_rotation(wrist, index, middle):
    global WRIST_ROTATION

    dir1 = index - wrist
    dir2 = middle - wrist

    print(norm(dir1), norm(dir2))
    second = norm(torch.cross(dir2, dir1))
    prim = norm(2 * dir2 + dir1)
    forward = norm(torch.cross(prim, second))
    up = second

    wrist_rotation = norm(look_rotation(up, forward))
    print("hand direction", forward)
    print("wrist", wrist_rotation)
    WRIST_ROTATION = wrist_rotation


class BasisHelper:

    @staticmethod
    def init_joint_info(name, curr_rot, prim, second):
        forward = norm(torch.cross(second, prim))
        up = second

        desired_rot = look_rotation(forward, up)

        real_curr_rot = norm(quaternion_multiply(curr_rot, WRIST_ROTATION))

        print(name, forward, second, desired_rot, curr_rot, real_curr_rot)
        offset = norm(quaternion_multiply(quaternion_invert(desired_rot), curr_rot))
        # print(name, offset)
        return offset

    @staticmethod
    def init_finger_info(fc: FingerConfig, pos, rot):
        pos0: torch.Tensor = pos[fc.pos_index[0]]
        pos1: torch.Tensor = pos[fc.pos_index[1]]
        pos2: torch.Tensor = pos[fc.pos_index[2]]
        pos3: torch.Tensor = pos[fc.pos_index[3]]
        f0prim = norm(pos0 - pos[0])
        f1prim = norm(pos1 - pos0)
        f2prim = norm(pos2 - pos1)
        f3prim = norm(pos3 - pos2)

        rot_dir_0 = norm(torch.cross(f1prim, f0prim))
        rot_dir_1 = norm(torch.cross(f2prim, f1prim))
        rot_dir_2 = norm(torch.cross(f3prim, f2prim))

        print(fc.name + " rotation dir", rot_dir_1, rot_dir_2)

        f1second = norm(torch.cross(rot_dir_1, f1prim))
        f2second = norm(torch.cross(rot_dir_1, f2prim))
        f3second = norm(torch.cross(rot_dir_1, f3prim))

        offset1 = BasisHelper.init_joint_info(fc.name + "_1", rot[fc.rot_index[0]], f1prim, f1second)
        offset2 = BasisHelper.init_joint_info(fc.name + "_2", rot[fc.rot_index[1]], f2prim, f2second)
        offset3 = BasisHelper.init_joint_info(fc.name + "_3", rot[fc.rot_index[2]], f3prim, f3second)
        return torch.stack([offset1, offset2, offset3])

    @staticmethod
    def get_basis(rotation, position):
        """
        rotation with hands_mean
        Args:
            rotation:
            position:

        Returns:

        """
        # print(position)
        calc_hand_rotation(position[0], position[1], position[4])
        # quaternion in w,x,y,z order
        rot_quat = axis_angle_to_quaternion(rotation)
        rot_quat = convert_local_to_global(rot_quat)
        # print(rot_quat)
        res = [BasisHelper.init_finger_info(INDEX, position, rot_quat),
               BasisHelper.init_finger_info(MIDDLE, position, rot_quat),
               BasisHelper.init_finger_info(PINKY, position, rot_quat),
               BasisHelper.init_finger_info(RING, position, rot_quat),
               BasisHelper.init_finger_info(THUMB, position, rot_quat)]

        res = torch.concat(res)
        return res

        # print(rotation.shape, position.shape)
        # print(rotation, position)


def test_look_at():
    tmp = torch.tensor([1., 1., 1.])
    forward = torch.tensor([-1., 0.5, 0.25])
    up = torch.cross(forward, tmp)

    # forward = torch.tensor([0, 0, 1.])
    # up = torch.tensor([0, 1., 0])
    rot = look_rotation(forward, up)
    print(forward)
    print(up)
    print(rot)


if __name__ == "__main__":
    test_look_at()
    # bh.axis_angle_to_quaternion(aa)


