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
from parse_xml import parseXML
from input_velodyne import *
import matplotlib.pyplot as plt

def convert_xyz_to_2d(places):
    theta = np.arctan2(places[:, 1], places[:, 0])
    ave_theta = np.average(theta)
    phi = np.arctan2(places[:, 2], np.sqrt(places[:, 0]**2 + places[:, 1]**2))
    ave_phi = np.average(phi)
    r = (theta / ave_theta).astype(np.int32)
    c = (phi / ave_phi).astype(np.int32)
    print "places", places.shape
    print np.max(places, axis=0)
    print np.min(places, axis=0)
    print "theta", theta.shape
    print theta.max(axis=0)
    print theta.min(axis=0)
    print ave_theta
    print "phi", phi.shape
    print phi.min()
    print phi.max()
    print ave_phi
    print r.max(), r.min(), c.max(), c.min()
    plt.hist(phi)
    plt.show()

def bird_view(places):
    pass

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


    pcd_path = "/home/katou01/download/training/velodyne/005080.bin"
    label_path = "/home/katou01/download/training/label_2/005080.txt"
    calib_path = "/home/katou01/download/training/calib/005080.txt"
    process(pcd_path, label_path, calib_path=calib_path, dataformat="bin", is_velo_cam=True)
