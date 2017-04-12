#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

"""                             0.1      0.2    0.5
Z  [-4.5, 5.5]    [ -45,  55]   100       50     20
Y  [ -50, 50.]    [-500, 500]   1000     500    200
X  [   0, 90.]    [   0, 900]   900      450    180

resolution 0.1, 0.2 0.5で試してみる
channel数は、1  np.newaxisで追加する
batch 10くらい

tensorflow
"""

def raw_to_voxel(pc, resolution=0.50):
    logic_x = np.logical_and(pc[:, 0] >= 0, pc[:, 0] <90)
    logic_y = np.logical_and(pc[:, 1] >= -50, pc[:, 1] < 50)
    logic_z = np.logical_and(pc[:, 2] >= -4.5, pc[:, 2] < 5.5)
    pc = pc[:, :3][np.logical_and(logic_x, np.logical_and(logic_y, logic_z))]
    pc =((pc - np.array([0., -50., -4.5])) / resolution).astype(np.int32)
    voxel = np.zeros((int(90 / resolution), int(100 / resolution), int(10 / resolution)))
    voxel[pc[:, 0], pc[:, 1], pc[:, 2]] = 1
    print voxel.shape
    return voxel

def center_to_sphere(places, size, resolution=0.50):
    """from label center to sphere center"""
    # for 1/4 sphere
    center = places.copy()
    center[:, 0] = center[:, 0] + size[:, 0] / 2.
    sphere_center = ((center - np.array([0., -50., -4.5])) / (resolution * 4)).astype(np.int32)
    return sphere_center

def sphere_to_center(p_sphere, resolution):
    """from sphere center to label center"""
    center = p_sphere * (resolution*4) + np.array([0., -50., -4.5])
    return center

def voxel_to_corner(corner_vox, resolution, center):#TODO
    """from """
    corners = center + corner_vox
    return corners

def corner_to_train(corners, sphere_center, resolution=0.50):
    """compare original corners  and  sphere centers"""
    sphere_center = sphere_to_center(sphere_center, resolution)
    for index, (corner, center) in enumerate(zip(corners, sphere_center)):
        corners[index] = corner - center
    return corners

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
        # places[:, 2] += size[:, 2] / 2.
        corners = get_boxcorners(places, rotates, size)
        a = corners.reshape(-1, 3).max(axis=0)
        max_val[max_val < a] = a[max_val < a]
        a = corners.reshape(-1, 3).min(axis=0)
        min_val[min_val > a] = a[min_val > a]

    print "max_val"
    print max_val
    print "min_val"
    print min_val
    print "finished"

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

    corners = get_boxcorners(places, rotates, size)
    filter_car_data(corners)
    pc = filter_camera_angle(pc)


    voxel =  raw_to_voxel(pc)
    center_sphere = center_to_sphere(places, size)
    corner_label = corner_to_train(corners, center_sphere)
    print center_sphere
    print corner_label


    # print pc.shape
    # print pc.max(axis=0), pc.min(axis=0)
    # print pc
    # publish_pc2(pc, obj)
    # publish_pc2(pc, corners.reshape(-1, 3))

if __name__ == "__main__":
    # pcd_path = "../data/training/velodyne/000012.pcd"
    # label_path = "../data/training/label_2/000012.txt"
    # calib_path = "../data/training/calib/000012.txt"
    # process(pcd_path, label_path, calib_path=calib_path, dataformat="pcd")

    # bin_path = "../data/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000030.bin"
    # xml_path = "../data/2011_09_26/2011_09_26_drive_0001_sync/tracklet_labels.xml"
    # process(bin_path, xml_path, dataformat="bin", label_type="xml")

    pcd_path = "/home/katou01/download/training/velodyne/000700.bin"
    label_path = "/home/katou01/download/training/label_2/000700.txt"
    calib_path = "/home/katou01/download/training/calib/000700.txt"
    process(pcd_path, label_path, calib_path=calib_path, dataformat="bin", is_velo_cam=True)
