import os.path
import pickle

from reprojection import generate_2d
from retargeting.basis import *
from retargeting.mesh_utils import *
from utils.mano_wrist import rot_pose_beta_to_mesh, hands_mean
from pytorch3d.transforms import quaternion_to_axis_angle
import xlrd

from pytorch3d.io import load_obj, save_obj
obj_filename = os.path.join('./data/hand_01.obj')
# obj_filename = os.path.join('extra_data/hand_mesh/hand.obj')
verts, faces, aux = load_obj(
            obj_filename,
            device="cuda",
            load_textures=True,
            create_texture_atlas=True,
            texture_atlas_size=8,
            texture_wrap=None,
        )
FACES = faces[0]

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

UNITY_BASIS = torch.tensor([[0.98252,0.1861579, 8.797169E-08, -1.925538E-07],[0.9867496,0.1622507, 8.623259E-07, -5.569176E-06],[0.998917,0.04652741, -3.814051E-07, -8.645397E-07],[0.9845469,0.1751214, -6.773092E-08, -1.296573E-07],[0.9895474,0.1442084, 2.347626E-07, 4.719131E-08],[0.9987367,0.0502522, 3.299475E-07, 2.839107E-07],[0.9802657,0.1976843, -4.922589E-09, -2.051222E-07],[0.9878246,0.1555727, -1.964481E-07, 5.758296E-08],[0.9954529,0.09525843, 4.293424E-07, 1.511594E-06],[0.9996653,0.02587354, -1.435634E-07, 1.650499E-07],[0.9994696,0.03256664, 2.490357E-07, -3.985346E-08],[0.9999212,0.01256056, -1.017585E-07, -2.349624E-09],[-0.9291456,-0.3697141, 2.073295E-07, -2.429503E-07],[-0.9564937,-0.2917534, 3.610364E-06, -1.518497E-05],[-0.9723085,-0.233702, 2.076902E-08, -6.052665E-07]], dtype=torch.float, device=device)
MANO_BASIS = torch.tensor([[ 0.9973, -0.0215, -0.0324,  0.0616],
        [ 0.9993,  0.0230,  0.0032, -0.0292],
        [ 0.9992, -0.0185,  0.0262, -0.0252],
        [ 0.9953, -0.0267,  0.0893,  0.0271],
        [ 0.9955, -0.0322,  0.0842, -0.0297],
        [ 0.9950, -0.0078,  0.0995,  0.0057],
        [ 0.9486,  0.0659,  0.3088,  0.0216],
        [ 0.9500,  0.1272,  0.2845,  0.0189],
        [ 0.9690,  0.0084,  0.2448, -0.0313],
        [ 0.9954, -0.0097,  0.0891,  0.0351],
        [ 0.9843, -0.0753,  0.1529, -0.0463],
        [ 0.9850, -0.0863,  0.1479, -0.0226],
        [ 0.5461, -0.7217, -0.2437,  0.3487],
        [ 0.3954, -0.8682, -0.2121,  0.2121],
        [ 0.5029, -0.8037, -0.2171,  0.2324]], dtype=torch.float, device=device)


def coordinate_change(data: torch.Tensor):
    """
    Convert data between left hand coordinate to rirght hand coordinate.
    Switch y and z, w = -w.
    Args:
        data:

    Returns:

    """
    if len(data.shape) == 2:
        data = data[:, [0, 1, 3, 2]]
        data[:, 0] *= -1
    else:
        assert len(data.shape) == 1
        data = data[[0, 1, 3, 2]]
        data[0] *= -1
    return data


def get_unity_root_based_rotation(data: torch.Tensor):
    data_index = [0, 17, 18, 19, 12, 13, 14, 2, 3, 4, 7, 8, 9, 22, 23, 24]
    data = data[data_index]
    global_rotation = data[0]
    data = quaternion_multiply(quaternion_invert(global_rotation), data)
    data[1:] = quaternion_multiply(quaternion_invert(UNITY_BASIS), data[1:])
    return global_rotation, data


def get_root_aa(root_rot):
    """
    Get root based axis-angle of mano based on the data exported from unity
    Args:
        root_rot:

    Returns:

    """
    # Rotation between mano and unity when root has no rotation
    mano_to_unity_quat = torch.tensor([0.9928, 0.0000, 0.1160, -0.0300], dtype=torch.float, device=device)
    mano_to_unity_quat = coordinate_change(mano_to_unity_quat)

    quat = root_rot
    quat = coordinate_change(quat)
    quat = quaternion_multiply(quat, mano_to_unity_quat)

    rot = torch.tensor([0.7071, -0.7071, 0, 0], dtype=torch.float, device=device)
    quat = quaternion_multiply(rot, quat)

    aa = quaternion_to_axis_angle(quat)

    return aa


def get_mano_parameter(global_rotation: torch.Tensor, unity_rot_tensor: torch.Tensor):
    """
    Get rot_pose_beta_to_mesh of mano based on the exported data from unity
    Args:
        global_rotation:
        unity_rot_tensor:

    Returns:

    """
    unity_rot_tensor = coordinate_change(unity_rot_tensor)

    unity_rot_tensor[1:] = quaternion_multiply(unity_rot_tensor[1:], MANO_BASIS)
    # print(unity_rot_tensor)
    unity_local_rot_tensor = convert_global_to_local(unity_rot_tensor)
    # print(unity_local_rot_tensor)
    pose_aa = quaternion_to_axis_angle(unity_local_rot_tensor)
    root_rot = get_root_aa(global_rotation)
    return root_rot, pose_aa


def generate_single_mesh(root_rot, pose, name):
    """
    Generate mesh in target file path
    Args:
        root_rot: rotation of root
        pose: rotation of joints
        name: target file path for mesh

    Returns:

    """
    all_data = rot_pose_beta_to_mesh(root_rot.view(1, 3), pose.contiguous().view(1, 45))
    mesh_data = all_data[0, 21:]
    joint_position = all_data[0, :21]
    # print("joint position", joint_position)
    # print(mesh_data.shape)

    # If render image is not needed, the next two lines can be skiped
    modify_vertices(mesh_data)  # generate obj with vt and vn
    render_mesh_image()  # render image

    if name != "":
        save_obj(name, mesh_data, FACES)
    save_obj('test/from_unity.obj', mesh_data, FACES)
    return joint_position


def generate_obj(dir_path):
    """

    Args:
        dir_path:

    Returns:

    """
    excel_path = dir_path + "/ExcelData.xls"
    # rotation_data_index = 5
    rotation_data_index = 4
    dir_name = os.path.dirname(excel_path)
    unity_rot = []
    with xlrd.open_workbook(excel_path) as wb:
        # print(wb)
        sheet = wb.sheet_by_index(0)
        idx = 0
        for row in sheet.get_rows():
            if 1 <= idx <= 26:
                unity_rot.append(torch.tensor([float(x) for x in row[rotation_data_index].value.split(",")], dtype=torch.float, device=device))
            idx += 1
    unity_origin_tensor = torch.stack(unity_rot)
    # print(unity_origin_tensor)
    root_rot, pose_rot = get_unity_root_based_rotation(unity_origin_tensor)
    root_aa, pose_aa = get_mano_parameter(root_rot, pose_rot)
    pose_aa -= hands_mean.view(pose_aa.shape)
    obj_name = dir_name + "/mano_mesh.obj"
    joint_3d = generate_single_mesh(root_aa, pose_aa, obj_name)
    joint_2d = generate_2d(dir_path)
    output_data = {
        "root": root_aa,
        "pose": pose_aa,
        "joint_3d": joint_3d,
        "joint_2d": joint_2d
    }
    pickle.dump(output_data, open(dir_name + "/output.pkl", 'wb'))


def test_load_data():
    data = pickle.load(open("output.pkl", 'rb'))
    generate_single_mesh(data["root"], data["pose"], "")


if __name__ == '__main__':
    excel_path = r"C:\Users\FengWang\Downloads\newhand\Hand\Build_Hand\ExportData\2022-08-02_17-22-21"
    generate_obj(excel_path)
