import numpy as np
from skspatial.objects import Plane, Points
import yaml

'''
    L - LiDAR 
    C - Camera
    G - Ground
    Translation vec in [m] 
    Given:
        - A_L_C - transformation mat. from L to C
        - h_L   - height of L w.r.t. G
    Want:
        - A_G_C - transformation mat. from G to C
'''

def load_proj_mat(yaml_path):
    with open(yaml_path) as stream:
        try:
            project_matrix = yaml.safe_load(stream)['extrinsic_matrix']['data'][:12]
            project_matrix = np.asarray(project_matrix).reshape(1, 1, 3, 4).tolist()
        except yaml.YAMLError as exc:
            print(exc)
    return project_matrix


def read_bin(bin_path, keep_intensity=False):
    OS1_POINTCLOUD_SHAPE = [1024, 128, 3]
    num_points = OS1_POINTCLOUD_SHAPE[0]*OS1_POINTCLOUD_SHAPE[1]
    bin_np = np.fromfile(bin_path, dtype=np.float32).reshape(num_points, -1)
    if not keep_intensity:
        bin_np = bin_np[:, :3]
    return bin_np



def get_norm_vec(path):
    pcs = read_bin(path)
    pcs = pcs[(pcs[:, 2] < -1.2) & (pcs[:, 2] > -3)]
    points = Points(pcs)
    plane = Plane.best_fit(points)
    normal = np.array(plane.normal)
    return normal

def get_rot_mat_from_norm_vec(n):
    """
    Source:
        https://math.stackexchange.com/questions/1956699/getting-a-transformation-matrix-from-a-normal-vector/1956758#1956758
    
    Computes the rotation matrix R for a given 3x1 vector n.

    Parameters:
    n (np.array): A numpy array of shape (3,) representing the vector (nx, ny, nz).

    Returns:
    np.array: A 3x3 numpy array representing the rotation matrix R.
    """
    
    # Ensure that the input is a numpy array
    n = np.asarray(n)
    
    # Extract components of the vector
    nx, ny, nz = n

    # Calculate the elements of the rotation matrix
    denominator = np.sqrt(nx**2 + ny**2)
    if denominator == 0:
        # The vector is aligned with the z-axis
        if nz == 0:
            # The vector is zero, return identity matrix
            return np.eye(3)
        else:
            # The vector is on the z-axis, no rotation around z-axis needed
            return np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, np.sign(nz)]
            ])
    
    u1 = ny / denominator
    u2 = -nx / denominator
    u3 = 0
    v1 = nx * nz / denominator
    v2 = ny * nz / denominator
    v3 = -denominator
    w1 = nx
    w2 = ny
    w3 = nz

    # Construct the rotation matrix
    R = np.array([
        [u1, u2, u3],
        [v1, v2, v3],
        [w1, w2, w3]
    ])
    
    return R.T


def main():

    yaml_path = '/robodata/ecocar_logs/processed/CACCDataset/calibrations/44/calib_os1_to_cam0.yaml'
    # load transformation mat. from L to C
    A_L_C = np.eye(4, dtype=float)
    A_L_C[:3, :] = np.array(load_proj_mat(yaml_path), dtype=float)[0, 0, :, :]
    R_L_C = A_L_C[:3, :3]

    # assumption that L was not tilted
    R_G_L = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    h_L   =  np.array([0, 0, 1.2])

    # R_G_C = R_L_C @ R_G_L
    path = "/robodata/ecocar_logs/processed/CACCDataset/3d_raw/os1/44/3d_raw_os1_44_1.bin"
    normal = get_norm_vec(path)
    R_G_L = get_rot_mat_from_norm_vec(normal) # similar to 90 CCW direction z-direction
    
    import pdb; pdb.set_trace()
    A_G_C = np.eye(4, dtype=float)
    A_G_C[:3, :3] = R_G_C
    A_G_C[:3, -1] = h_L

    return A_G_C

if __name__== "__main__":
    main()