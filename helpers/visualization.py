import os
import json

#Libraries
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.cm as cm

import cv2
from cv_bridge import CvBridge

#ROS
import rospy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2, Imu
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped
import sensor_msgs.point_cloud2 as pc2

#Custom
# from helpers.constants import *
# from helpers.sensors import read_sem_label
# from helpers.geometry import *
# from helpers.calibration import load_extrinsic_matrix


def get_coeff(x, y, deg=3):
    """Return coefficients of a fitted polynomial.

    Keyword arguments:
    x -- array of x values
    y -- array of y values
    """

    coeff = np.polyfit(x, y, deg)
    coeff = np.flip(coeff)
    
    return coeff

def get_homo(trans, yaw):
    
    rot_mat = R.from_euler('z', yaw, degrees=True).as_matrix()

    homo_mat = np.eye(4, dtype=np.float64)
    homo_mat[:3, :3] = rot_mat
    homo_mat[:3, 3] = trans
    return homo_mat

def transform_gnd_to_gm(P_GND):
    """Return pointcloud in GM coordinate.

    Keyword arguments:
    P_GND -- the raw res from Anchord3DLane model in ground reference frame.
    """

    T_LIDAR_GND = get_homo(trans=np.array([0, 0, 5]).reshap(-1, 1), yaw=0) # np.eye(4, dtype=np.float64)
    T_FCM_LIDAR = get_homo(trans=np.array([0, 0, 5]).reshap(-1, 1), yaw=45) # np.eye(4, dtype=np.float64)
    T_GM_FCM = get_homo(trans=np.array([0, 0, 0]).reshap(-1, 1), yaw=90) 

    pixel = 10
    meters = 10
    ALPHA = pixel/meters

    P_GM = ALPHA @ T_GM_FCM @ T_FCM_LIDAR @ T_LIDAR_GND @ P_GND

    return P_GM


def pred2lanes(pred):
    anchor_y_steps = [5,  10,  15,  20,  30,  40,  50,  60,  80,  100]
    anchor_len = len(anchor_y_steps)
    ys = np.array(anchor_y_steps, dtype=np.float32)
    lanes = []
    probs = []
    for lane in pred:
        import pdb; pdb.set_trace()
        lane_xs = lane[5:5 + anchor_len]
        lane_zs = lane[5 + anchor_len : 5 + 2 * anchor_len]
        lane_vis = (lane[5 + anchor_len * 2 : 5 + 3 * anchor_len]) > 0
        if (lane_vis).sum() < 2:
            continue
        lane_xs = lane_xs[lane_vis]
        lane_ys = ys[lane_vis]
        lane_zs = lane_zs[lane_vis]
        lanes.append(np.stack([lane_xs, lane_ys, lane_zs], axis=-1).tolist())
        probs.append(float(lane[-1]))

    return lanes, probs

def pred2apollosimformat(pred, proposal_key = 'proposals_list'):
    line = dict()
    pred_proposal = pred[proposal_key]
    pred_lanes, prob_lanes = pred2lanes(pred_proposal)
    line["laneLines"] = pred_lanes
    line["laneLines_prob"]  = prob_lanes
    return line

def extract_line(prediction):
    line = pred2apollosimformat(prediction)