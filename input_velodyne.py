#!/usr/bin/env python
import sys
import os
import rospy
import numpy as np
import cv2
import pcl
import glob
import math
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
# from cv_bridge import CvBridge
from parse_xml import parseXML

"""
TODO
1. Only get Car Label and Data
2. get by batch data and label
3. learning by 3D CNN
4. nms non maximus suppression

"""
def load_pc_from_pcd(pcd_path):
    p = pcl.load(pcd_path)
    return np.array(list(p), dtype=np.float32)

def load_pc_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

def load_pc_from_bins(bin_dir, batch_size, shuffle=False): #TODO: Generator for 3D data
    bin_pathes = glob.glob(bin_dir + "/*.bin").sort()
    data_size = len(bin_pathes)
    bathes = None

    if shuffle:
        perm = np.random.permutation(data_size)
        batches = [perm[i * batch_size:(i + 1) * batch_size]]
    else:
        perm = np.arange(data_size)
        batches = [perm[i * batch_size:(i + 1) * batch_size] for i in range(int(np.ceil(data_size / batch_size)))]

    imgfiles = [[bin_pathes[p] for p in b] for b in batches]

    # imgs = ImageLoader(imgfiles)

    for p, imgs in itertools.izip(batches, imgs.wait_images()):
        for index, img in enumerate(imgs):
            imgs[index] = np.array(img, dtype=np.float32)
        if self._callback is not None:
            imgs = self._callback.create(np.array(imgs, dtype=np.float32))
        if self._data_y == None:
            yield np.array(imgs, dtype=np.float32)
        yield np.array(imgs, dtype=np.float32), self._data_y[p]

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

            if label[0] == ("Car" or "Van"): #  or "Truck"
                bounding_box.append(label[8:15])

    if bounding_box:
        data = np.array(bounding_box, dtype=np.float32)
        return data[:, 3:6], data[:, :3], data[:, 6]
    else:
        return None, None, None

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
    rect = calib_data["R0_rect"].reshape(3, 3)
    velo_to_cam = calib_data["Tr_velo_to_cam"].reshape(3, 4)
    inv_rect = np.linalg.inv(rect)
    inv_velo_to_cam = np.linalg.pinv(velo_to_cam[:, :3])
    return np.dot(inv_velo_to_cam, inv_rect)

def filter_camera_angle(places):
    bool_in = np.logical_and((places[:, 1] < places[:, 0] - 0.27), (-places[:, 1] < places[:, 0] - 0.27))
    # bool_in = np.logical_and((places[:, 1] < places[:, 0]), (-places[:, 1] < places[:, 0]))
    return places[bool_in]

def filter_car_data(corners):
    # consider bottom square
    # print corners[0]
    # argcor = np.argsort(corners[0], axis=0)
    # print corners[0].shape
    # print corners[0][argcor]
    pass

def create_publish_obj(obj, places, rotates, size):
    for place, rotate, sz in zip(places, rotates, size):
        x, y, z = place
        # print (x, y, z)
        obj.append((x, y, z))
        h, w, l = sz
        # print h, w, l
        if l > 10:
            continue
        for hei in range(0, int(h*100)):
            for wid in range(0, int(w*100)):
                for le in range(0, int(l*100)):
                    a = (x - l / 2.) + le / 100.
                    b = (y - w / 2.) + wid / 100.
                    c = (z) + hei / 100.
                    obj.append((a, b, c))
    return obj

def get_boxcorners(places, rotates, size):
    corners = []
    for place, rotate, sz in zip(places, rotates, size):
        x, y, z = place
        h, w, l = sz
        if l > 10:
            continue

        corner = np.array([
            [x - l / 2., y - w / 2., z],
            [x + l / 2., y - w / 2., z],
            [x - l / 2., y + w / 2., z],
            [x - l / 2., y - w / 2., z + h],
            [x - l / 2., y + w / 2., z + h],
            [x + l / 2., y + w / 2., z],
            [x + l / 2., y - w / 2., z + h],
            [x + l / 2., y + w / 2., z + h],
        ])

        corner -= np.array([x, y, z])

        rotate_matrix = np.array([
            [np.cos(rotate), -np.sin(rotate), 0],
            [np.sin(rotate), np.cos(rotate), 0],
            [0, 0, 1]
        ])

        a = np.dot(corner, rotate_matrix.transpose())
        a += np.array([x, y, z])
        corners.append(a)
    return np.array(corners)

def read_labels(label_path, label_type, calib_path=None, is_velo_cam=False, proj_velo=None):
    if label_type == "txt": #TODO
        places, size, rotates = read_label_from_txt(label_path)
        if places is None:
            return None, None, None
        rotates = np.pi / 2 - rotates
        dummy = np.zeros_like(places)
        dummy = places.copy()
        if calib_path:
            places = np.dot(dummy, proj_velo.transpose())[:, :3]
        else:
            places = dummy
        if is_velo_cam:
            places[:, 0] += 0.27

    elif label_type == "xml":
        bounding_boxes, size = read_label_from_xml(label_path)
        places = bounding_boxes[30]["place"]
        rotates = bounding_boxes[30]["rotate"][:, 2]
        size = bounding_boxes[30]["size"]

    return places, rotates, size

def publish_pc2(pc, obj):
    pub = rospy.Publisher("/points_raw", PointCloud2, queue_size=1000000)
    rospy.init_node("pc2_publisher")
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "velodyne"
    points = pc2.create_cloud_xyz32(header, pc[:, :3])

    pub2 = rospy.Publisher("/points_raw1", PointCloud2, queue_size=1000000)
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "velodyne"
    points2 = pc2.create_cloud_xyz32(header, obj)

    r = rospy.Rate(0.1)
    while not rospy.is_shutdown():
        pub.publish(points)
        pub2.publish(points2)
        r.sleep()

def raw_to_voxel(pc, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5)):
    logic_x = np.logical_and(pc[:, 0] >= x[0], pc[:, 0] < x[1])
    logic_y = np.logical_and(pc[:, 1] >= y[0], pc[:, 1] < y[1])
    logic_z = np.logical_and(pc[:, 2] >= z[0], pc[:, 2] < z[1])
    pc = pc[:, :3][np.logical_and(logic_x, np.logical_and(logic_y, logic_z))]
    pc =((pc - np.array([x[0], y[0], z[0]])) / resolution).astype(np.int32)
    voxel = np.zeros((int((x[1] - x[0]) / resolution), int((y[1] - y[0]) / resolution), int(round((z[1]-z[0]) / resolution))))
    voxel[pc[:, 0], pc[:, 1], pc[:, 2]] = 1
    return voxel

def center_to_sphere(places, size, resolution=0.50, min_value=np.array([0., -50., -4.5]), scale=4, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5)):
    """from label center to sphere center"""
    # for 1/4 sphere
    x_logical = np.logical_and((places[:, 0] < x[1]), (places[:, 0] >= x[0]))
    y_logical = np.logical_and((places[:, 1] < y[1]), (places[:, 1] >= y[0]))
    z_logical = np.logical_and((places[:, 2] < z[1]), (places[:, 2] >= z[0]))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
    center = places.copy()
    center[:, 2] = center[:, 2] + size[:, 0] / 2.
    sphere_center = ((center[xyz_logical] - min_value) / (resolution * scale)).astype(np.int32)
    return sphere_center

def sphere_to_center(p_sphere, resolution=0.5, scale=4, min_value=np.array([0., -50., -4.5])):
    """from sphere center to label center"""
    center = p_sphere * (resolution*scale) + min_value
    return center

def voxel_to_corner(corner_vox, resolution, center):#TODO
    """from """
    corners = center + corner_vox
    return corners

def create_label(places, size, corners, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5), scale=4, min_value=np.array([0., -50., -4.5])):
    x_logical = np.logical_and((places[:, 0] < x[1]), (places[:, 0] >= x[0]))
    y_logical = np.logical_and((places[:, 1] < y[1]), (places[:, 1] >= y[0]))
    z_logical = np.logical_and((places[:, 2] + size[:, 0]/2. < z[1]), (places[:, 2] + size[:, 0]/2. >= z[0]))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
    center = places.copy()
    center[:, 2] = center[:, 2] + size[:, 0] / 2.
    sphere_center = ((center[xyz_logical] - min_value) / (resolution * scale)).astype(np.int32)

    train_corners = corners[xyz_logical].copy()
    anchor_center = sphere_to_center(sphere_center, resolution=resolution, scale=scale, min_value=min_value) #sphere to center
    for index, (corner, center) in enumerate(zip(corners[xyz_logical], anchor_center)):
        train_corners[index] = corner - center
    return sphere_center, train_corners

def corner_to_train(corners, sphere_center, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5), scale=4, min_value=np.array([0., -50., -4.5])):
    """compare original corners  and  sphere centers"""
    x_logical = np.logical_and((corners[:, :, 0] < x[1]), (corners[:, :, 0] >= x[0]))
    y_logical = np.logical_and((corners[:, :, 1] < y[1]), (corners[:, :, 1] >= y[0]))
    z_logical = np.logical_and((corners[:, :, 2] < z[1]), (corners[:, :, 2] >= z[0]))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical)).all(axis=1)
    train_corners = corners[xyz_logical].copy()
    sphere_center = sphere_to_center(sphere_center, resolution=resolution, scale=scale, min_value=min_value) #sphere to center
    for index, (corner, center) in enumerate(zip(corners[xyz_logical], sphere_center)):
        train_corners[index] = corner - center
    return train_corners

def corner_to_voxel(voxel_shape, corners, sphere_center, scale=4):
    corner_voxel = np.zeros((voxel_shape[0] / scale, voxel_shape[1] / scale, voxel_shape[2] / scale, 24))
    corner_voxel[sphere_center[:, 0], sphere_center[:, 1], sphere_center[:, 2]] = corners
    return corner_voxel

def create_objectness_label(sphere_center, resolution=0.5, x=90, y=100, z=10, scale=4):
    obj_maps = np.zeros((int(x / (resolution * scale)), int(y / (resolution * scale)), int(round(z / (resolution * scale)))))
    obj_maps[sphere_center[:, 0], sphere_center[:, 1], sphere_center[:, 2]] = 1
    return obj_maps

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
    pc = filter_camera_angle(pc)
    # obj = []
    # obj = create_publish_obj(obj, places, rotates, size)

    p.append((0, 0, 0))
    p.append((0, 0, -1))
    print pc.shape
    print 1
    # publish_pc2(pc, obj)
    a = center_to_sphere(places, size, resolution=0.25)
    print places
    print a
    print sphere_to_center(a, resolution=0.25)
    bbox = sphere_to_center(a, resolution=0.25)
    print corners.shape
    # a = np.array(
    #     [[ 19.69109106, 8.70038319, -2.05356455],
    #     [ 18.27717495, 5.61360097, -1.26570401],
    #     [ 21.56218159, 8.64504647, -1.58700204],
    #     [ 20.02987021, 8.84583879, -0.10549831],
    #     [ 20.75653511, 8.11167407, 0.5703392 ],
    #     [ 19.77633509, 5.56351113, -0.7579807 ],
    #     [ 19.72957426, 5.75904274, -0.37826872],
    #     [ 20.75458926, 5.8138907 , -0.41885149]]
    # )
    # print a.shape
    # publish_pc2(pc, bbox.reshape(-1, 3))
    publish_pc2(pc, corners.reshape(-1, 3))

if __name__ == "__main__":
    # pcd_path = "../data/training/velodyne/000012.pcd"
    # label_path = "../data/training/label_2/000012.txt"
    # calib_path = "../data/training/calib/000012.txt"
    # process(pcd_path, label_path, calib_path=calib_path, dataformat="pcd")

    # bin_path = "../data/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000030.bin"
    # xml_path = "../data/2011_09_26/2011_09_26_drive_0001_sync/tracklet_labels.xml"
    # process(bin_path, xml_path, dataformat="bin", label_type="xml")


    pcd_path = "../data/training/velodyne/000410.bin"
    label_path = "../data/training/label_2/000410.txt"
    calib_path = "../data/training/calib/000410.txt"
    process(pcd_path, label_path, calib_path=calib_path, dataformat="bin", is_velo_cam=True)
