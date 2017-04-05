#!/usr/bin/env python
import sys
import os
import rospy
import numpy as np
import cv2
import pcl
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge
from parse_xml import parseXML

def load_pc_from_pcd(pcd_path):
    p = pcl.load(pcd_path)
    return np.array(list(p), dtype=np.float32)

def load_pc_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

def read_label_from_txt(label_path):
    text = np.fromfile(label_path)
    bounding_box = []
    with open(label_path, "r") as f:
        labels = f.read().split("\n")
        for label in labels:
            if not label:
                continue
            label = label.split(" ")
            if (label[0] == "DontCare"):
                continue
            bounding_box.append(label[8:14])

    data = np.array(bounding_box, dtype=np.float32)
    return data[:, 3:], data[:, :3]

def read_label_from_xml(label_path):
    labels = parseXML(label_path)
    label_dic = {}
    for label in labels:
        first_frame = label.firstFrame
        nframes = label.nFrames
        size = label.size
        obj_type = label.objectType
        for index, place, rotate in zip(range(first_frame, first_frame+nframes), label.trans, label.rots):
            if index in label_dic.keys():
                label_dic[index]["place"] = np.vstack((label_dic[index]["place"], place))
                label_dic[index]["size"] = np.vstack((label_dic[index]["size"], np.array(size)))
                label_dic[index]["rotate"] = np.vstack((label_dic[index]["rotate"], rotate))
            else:
                label_dic[index] = {}
                label_dic[index]["place"] = place
                label_dic[index]["rotate"] = rotate
                label_dic[index]["size"] = np.array(size)

    return label_dic, size

def read_calib_file(calib_path):
    data = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if not line or line == "\n":
                continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def proj_to_velo(calib_data):
    p0 = calib_data['P0'].reshape(3, 4)
    velo_to_cam = calib_data["Tr_velo_to_cam"].reshape(3, 4)
    rect = calib_data["R0_rect"].reshape(3, 3)
    inv_rect = np.linalg.inv(rect)
    inv_p0 = np.linalg.inv(p0[:, :3])
    inv_velo_to_cam = np.linalg.pinv(velo_to_cam)
    # np.dot(cam_to_velo, np.dot(inv_rect,box[3:]))[:3]
    # return np.dot(inv_velo_to_cam, np.dot(inv_rect, inv_p0))
    return np.dot(inv_velo_to_cam, inv_rect)


def publish_pc2(velodyne_path, label_path=None, calib_path=None, dataformat="pcd", label_type="txt"):
    p = []
    pc = None
    bounding_boxes = None
    places = None
    rotates = None
    proj_velo = None

    if dataformat == "bin":
        pc = load_pc_from_bin(velodyne_path)
    elif dataformat == "pcd":
        pc = load_pc_from_pcd(velodyne_path)

    if calib_path:
        calib = read_calib_file(calib_path)
        proj_velo = proj_to_velo(calib)[:, :3]
        print proj_velo.shape

    obj = []
    if label_path:
        if label_type == "txt": #TODO
            places, size = read_label_from_txt(label_path)
            dummy = np.zeros_like(places)
            print places
            dummy = places.copy()
            # dummy[:, 0] = places[:, 2]
            # dummy[:, 1] = places[:, 0]
            # dummy[:, 2] = places[:, 1]
            if calib_path:
                print dummy.shape
                print "proj_velo"
                print proj_velo
                # places = np.dot(dummy.transpose(), proj_velo)
                places = np.dot(dummy, proj_velo.transpose())[:, :3]
                print places.shape
            else:
                places = dummy
            rotates = places #TODO

        elif label_type == "xml":
            bounding_boxes, size = read_label_from_xml(label_path)
            places = bounding_boxes[30]["place"]
            rotates = bounding_boxes[30]["rotate"]
            size = bounding_boxes[30]["size"]

    for place, rotate, sz in zip(places, rotates, size):
        x, y, z = place
        print (x, y, z)
        obj.append((x, y, z))
        h, w, l = sz
        print h, w, l
        if l > 10:
            continue
        for hei in range(0, int(h*100)):
            for wid in range(0, int(w*100)):
                for le in range(0, int(l*100)):
                    a = (x - l / 2.) + le / 100.
                    b = (y - w / 2.) + wid / 100.
                    c = (z) + hei / 100.
                    obj.append((a, b, c))

    p.append((0, 0, 0))
    print 1
    pub = rospy.Publisher("/points_raw", PointCloud2, queue_size=10000)
    rospy.init_node("pc2_publisher")
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "velodyne"
    points = pc2.create_cloud_xyz32(header, pc[:, :3])

    pub2 = rospy.Publisher("/points_raw1", PointCloud2, queue_size=10000)
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "velodyne"
    points2 = pc2.create_cloud_xyz32(header, obj)

    r = rospy.Rate(0.1)
    while not rospy.is_shutdown():
        pub.publish(points)
        pub2.publish(points2)
        r.sleep()

if __name__ == "__main__":
    pcd_path = "/home/katou01/download/training/velodyne/000010.pcd"
    label_path = "/home/katou01/download/training/label_2/000010.txt"
    calib_path = "/home/katou01/download/training/calib/000010.txt"
    publish_pc2(pcd_path, label_path, calib_path=calib_path, dataformat="pcd")

    # bin_path = "/home/katou01/download/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000030.bin"
    # xml_path = "/home/katou01/download/2011_09_26/2011_09_26_drive_0001_sync/tracklet_labels.xml"
    # publish_pc2(bin_path, xml_path, dataformat="bin", label_type="xml")
