import sys
import numpy as np
import tensorflow as tf
from input_velodyne import *

def train1(train_features, train_labels, valid_features, valid_labels, test_features, test_labels):
    # tf Graph input
    batch_size = 10
    training_epochs = 70

    test_accuracy_list = []
    valid_accuracy_list = []
    x, phase_train, y, optimizer, cost, accuracy =  build_graph(is_training=True)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        model = ssd_model(sess, voxel, voxel_shape=voxel.shape, activation=tf.nn.relu)
        total_loss = loss_func(model, center_sphere, g_map, g_cord)
        optimizer = create_optimizer(total_loss)

        for epoch in range(training_epochs):
            total_batch = int(len(train_features)/batch_size)
            for i in range(total_batch):
                batch_x, batch_y = train_features[i*batch_size : (i+1) * batch_size], train_labels[i*batch_size : (i+1) * batch_size]
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, phase_train: True})
                c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
            c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")
        saver.save(sess, "finish_YUV_100_8_378_00005.ckpt")

    return valid_accuracy_list, test_accuracy_list


def process(velodyne_path, label_path=None, calib_path=None, resolution=0.2, dataformat="pcd", label_type="txt", is_velo_cam=False):
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

    voxel =  raw_to_voxel(pc, resolution=resolution)
    center_sphere = center_to_sphere(places, size, resolution=resolution)
    corner_label = corner_to_train(corners, center_sphere, resolution=resolution)
    g_map = create_objectness_label(center_sphere, resolution=resolution)
    g_cord = corner_label.reshape(corner_label.shape[0], -1)

    voxel = voxel.reshape(1, voxel.shape[0], voxel.shape[1], voxel.shape[2], 1)
    g_map = g_map[np.newaxis,:, :, :]
    g_cord = g_cord[np.newaxis, :]
    center_sphere = center_sphere[np.newaxis, :]
    print voxel.shape

    with tf.Session() as sess:
        model = ssd_model(sess, voxel, voxel_shape=voxel.shape, activation=tf.nn.relu)
        print(vars(model))
        total_loss = loss_func(model, center_sphere, g_map, g_cord)
        optimizer = create_optimizer(total_loss)
