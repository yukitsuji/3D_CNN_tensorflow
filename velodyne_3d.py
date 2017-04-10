#!/usr/bin/env python

import sys
import os
import rospy
import numpy as np
import cv2
import pcl
import glob
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
# from cv_bridge import CvBridge
from parse_xml import parseXML
from input_velodyne import *
import matplotlib.pyplot as plt

def convert_xyz_to_2d(places):
    print places.shape
    print places.min(axis=0)
    print places.max(axis=0)
    print places.max(axis=0) - places.min(axis=0)
    plt.hist(places[:, 1])
    plt.show()

"""                             0.1      0.2     0.5
Z  [-2.5, 5.5]    [ -25,  55]   80       40      16
Y  [ -50, 50.]    [-500, 500]   1000     500    200
X  [   0, 90.]    [   0, 900]   900      450    180

resolution 0.1, 0.2 0.5で試してみる
channel数は、1  np.newaxisで追加する
batch 10くらい

tensorflow
"""

def confirm(velodyne_path, label_path=None, calib_path=None, dataformat="pcd", label_type="txt", is_velo_cam=False):
    p = []
    pc = None
    bounding_boxes = None
    places = None
    rotates = None
    size = None
    proj_velo = None

    calibs = glob.glob(calib_path)
    labels = glob.glob(label_path)
    max_val = np.array([0., 0., 0.])
    min_val = np.array([0., 0., 0.])
    for calib_path, label_path in zip(calibs, labels):
        calib = read_calib_file(calib_path)
        proj_velo = proj_to_velo(calib)[:, :3]
        places, rotates, size = read_labels(label_path, label_type, calib_path=calib_path, is_velo_cam=is_velo_cam, proj_velo=proj_velo)
        if places is None:
            continue
        places[:, 2] += size[:, 2] / 2.
        # corners = get_boxcorners(places, rotates, size)
        a = places.reshape(-1, 3).max(axis=0)
        max_val[max_val < a] = a[max_val < a]
        a = places.reshape(-1, 3).min(axis=0)
        min_val[min_val > a] = a[min_val > a]

    print "max_val"
    print max_val
    print "min_val"
    print min_val
    print "finished"

    #
    # # pc = filter_camera_angle(pc)
    # corners = get_boxcorners(places, rotates, size)
    # print corners
    # convert_xyz_to_2d(pc)


def process(velodyne_path, label_path=None, calib_path=None, dataformat="pcd", label_type="txt", is_velo_cam=False):
    p = []
    pc = None
    bounding_boxes = None
    places = None
    rotates = None
    size = None
    proj_velo = None

    if dataformat == "bin":
        pc = load_pc_from_bin(velodyne_path)
    elif dataformat == "pcd":
        pc = load_pc_from_pcd(velodyne_path)

    if calib_path:
        calib = read_calib_file(calib_path)
        proj_velo = proj_to_velo(calib)[:, :3]

    if label_path:
        places, rotates, size = read_labels(label_path, label_type, calib_path=calib_path, is_velo_cam=is_velo_cam, proj_velo=proj_velo)

    # pc = filter_camera_angle(pc)
    corners = get_boxcorners(places, rotates, size)
    print corners
    convert_xyz_to_2d(pc)

    # corners = get_boxcorners(places, rotates, size)
    # filter_car_data(corners)
    #
    # pc = filter_camera_angle(pc)
    # # obj = []
    # # obj = create_publish_obj(obj, places, rotates, size)
    #
    # p.append((0, 0, 0))
    # print 1
    # # publish_pc2(pc, obj)
    # publish_pc2(pc, corners.reshape(-1, 3))

if __name__ == "__main__":
    # pcd_path = "../data/training/velodyne/000012.pcd"
    # label_path = "../data/training/label_2/000012.txt"
    # calib_path = "../data/training/calib/000012.txt"
    # process(pcd_path, label_path, calib_path=calib_path, dataformat="pcd")

    # bin_path = "../data/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000030.bin"
    # xml_path = "../data/2011_09_26/2011_09_26_drive_0001_sync/tracklet_labels.xml"
    # process(bin_path, xml_path, dataformat="bin", label_type="xml")


    pcd_path = "../data/training/velodyne/000200.bin"
    label_path = "../data/training/label_2/*.txt"
    calib_path = "../data/training/calib/*.txt"
    confirm(pcd_path, label_path, calib_path=calib_path, dataformat="bin", is_velo_cam=True)
