import tensorflow as tf
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import pylab
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
"""
learning rate : 0.0005%
Y : 100, UV : 8
Batch Normalization
kernel_initialization = std*0.01
data normalization : data/255
"""
def build_graph(is_training):
    """define model architecture, loss function, optimizer"""
    def conv2d(x, W, b, stride=1):
        """define convolution layer"""
        x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        """define max pooling layer"""
        return tf.nn.max_pool(
            x,
            ksize = [1, k, k, 1],
            strides = [1, k, k, 1],
            padding='SAME')

    def batch_norm(inputs, is_training, decay=0.9, eps=1e-5):
        """Batch Normalization

           Args:
               inputs: input data(Batch size) from last layer
               is_training: when you test, please set is_training "None"
           Returns:
               output for next layer
        """
        gamma = tf.Variable(tf.ones(inputs.get_shape()[1:]), name="gamma")
        beta = tf.Variable(tf.zeros(inputs.get_shape()[1:]), name="beta")
        pop_mean = tf.Variable(tf.zeros(inputs.get_shape()[1:]), trainable=False, name="pop_mean")
        pop_var = tf.Variable(tf.ones(inputs.get_shape()[1:]), trainable=False, name="pop_var")

        if is_training != None:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean*(1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, eps)
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, eps)

    def create_model(x, weights, biases, is_training):
        """define model architecture"""
        conv1_1 = conv2d(tf.expand_dims(x[:, :, :, 0], 3), weights['layer_1_1'], biases['layer_1_1'])
        conv1_2 = conv2d(x[:, :, :, 1:], weights['layer_1_2'], biases['layer_1_2'])
        conv1 = tf.concat(3, [conv1_1, conv1_2])
        conv1 = maxpool2d(conv1, 2)
        conv1 = batch_norm(conv1, is_training)

        conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'])
        conv2 = maxpool2d(conv2, 2)
        conv2 = batch_norm(conv2, is_training)

        layer_3_1 = tf.reshape(
            conv2,
            [-1, 8*8*200]
        )
        layer_3_2 = tf.reshape(
            conv1,
            [-1, 16*16*108]
        )
        fc1 = tf.concat(1, [layer_3_1, layer_3_2])

        fully = tf.add(tf.matmul(fc1, weights['fully']), biases['fully'])
        fully = tf.nn.relu(fully)
        fully = batch_norm(fully, is_training)
        out = tf.add(tf.matmul(fully, weights['out']), biases['out'])
        return out

    layer_width = {
        'layer_1_1' : 100,
        'layer_1_2' : 8,
        'layer_2' : 200,
        'fully' : 300,
        'out' : 43
    }

    #weight =  [filter_width, filter_height, in_channels, out_channel]
    weights = {
        'layer_1_1' : tf.Variable(
            tf.truncated_normal([5, 5, 1, layer_width['layer_1_1']],
                stddev=0.01, seed=832289), name="w_layer_1_1"),
        'layer_1_2' : tf.Variable(
            tf.truncated_normal([5, 5, 2, layer_width['layer_1_2']],
                stddev=0.01, seed=832289), name="w_layer_1_2"),
        'layer_2' : tf.Variable(
            tf.truncated_normal([3, 3, layer_width['layer_1_1']+layer_width['layer_1_2'],
                layer_width['layer_2']], stddev=0.01, seed=832289), name="w_layer_2"),
        'fully' : tf.Variable(
            tf.truncated_normal([8 * 8 * layer_width['layer_2'] + 16*16*108, layer_width['fully']],
                stddev=0.01, seed=832289), name="w_fully"),
        'out' : tf.Variable(
            tf.truncated_normal([layer_width['fully'], layer_width['out']],
                stddev=0.01, seed=832289), name="w_out")
    }

    biases = {
        'layer_1_1' : tf.Variable(tf.zeros(layer_width['layer_1_1']), name="b_layer_1_1"),
        'layer_1_2' : tf.Variable(tf.zeros(layer_width['layer_1_2']), name="b_layer_1_2"),
        'layer_2' : tf.Variable(tf.zeros(layer_width['layer_2']), name="b_layer_2"),
        'fully' : tf.Variable(tf.zeros(layer_width['fully']), name="b_fully"),
        'out' : tf.Variable(tf.zeros(layer_width['out']), name="b_out")
    }


    x = tf.placeholder("float", [None, 32, 32, 3])
    y = tf.placeholder("float", [None, 43])
    phase_train = tf.placeholder(tf.bool, name='phase_train') if is_training else None

    classifier = create_model(x, weights, biases, phase_train)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(classifier, y))
    opt = tf.train.AdamOptimizer(0.0005)
    optimizer = opt.minimize(cost)
    correct_prediction = tf.equal(tf.argmax(classifier, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return x, phase_train, y, optimizer, cost, accuracy

def get_accuracy(x, y, phase_train, X_test, Y_test, accuracy, test_batch_size=30):
    """Get accuracy of selected datasets"""
    num_iter = X_test.shape[0] // test_batch_size
    num_accuracy= 0
    for ni in range(num_iter):
        num_accuracy += accuracy.eval({x : X_test[test_batch_size*ni : test_batch_size*(ni+1)],
                            y : Y_test[test_batch_size*ni : test_batch_size*(ni+1)], phase_train: None})
    num_accuracy = num_accuracy / num_iter
    return num_accuracy

def train_validation_test(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, training_epochs=200, batch_size=378):
    """Excecute Training, Validation, Test

       Returns:
           valid_accuracy_list (list): accuracy of Validation sets of each epoch
           test_accuracy_list (list): accuracy of Test sets of each epoch
    """
    batch_size = batch_size
    training_epochs = training_epochs

    test_accuracy_list = []
    valid_accuracy_list = []
    x, phase_train, y, optimizer, cost, accuracy =  build_graph(is_training=True)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(training_epochs):
            sum_cost = 0
            total_batch = len(X_train)//batch_size
            for i in range(total_batch):
                batch_x, batch_y = X_train[i*batch_size : (i+1) * batch_size], Y_train[i*batch_size : (i+1) * batch_size]
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, phase_train: True})
                sum_cost += sess.run(cost, feed_dict={x: X_train, y: Y_train})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(sum_cost))

            valid_accuracy = get_accuracy(x, y, phase_train, X_valid, Y_valid, accuracy, test_batch_size=30)
            valid_accuracy_list.append(valid_accuracy)
            print(
                "valid Accuracy:",
                valid_accuracy
            )

            test_accuracy = get_accuracy(x, y, phase_train, X_test, Y_test, accuracy, test_batch_size=30)
            test_accuracy_list.append(test_accuracy)
            print(
                "test Accuracy:",
                test_accuracy
            )
        print("Optimization Finished!")
        saver.save(sess, "model.ckpt")

    return valid_accuracy_list, test_accuracy_list

def shuffle_datasets(x, y):
    """shuffle datasets"""
    np.random.seed(832289)
    argnum = np.arange(y.shape[0])
    np.random.shuffle(argnum)
    x = x[argnum]
    y = y[argnum]
    return x, y

def RGB_to_YUV(images):
    """Image color space conversion from RGB to YUV

        Args:
            images (numpy array): 3 or 4 dimension. RGB Image
                                  4 dimension is (batch size, image shape)
        Returns
            images (numpy array): 3 or 4 dimension. YUV Image
    """
    if images.ndim == 4:
        for xi in range(images.shape[0]):
            images[xi] =  cv2.cvtColor(images[xi], cv2.COLOR_RGB2YUV)
        return images
    else:
        images = cv2.cvtColor(images, cv2.COLOR_RGB2YUV)
        return images

def divide_training_and_validataion(original_X_train, original_y_train, n_classes, test_size=0.08):
    """divide training sets into training and validation sets.
       Training Sets has same portion(0.92) of each category.
       Validation Sets has same portion(0.08)

       Args:
           original_X_train (numpy array): 4 dimension Datasets for Training(image)
           original_y_train (numpy array): 1-dimension Datasets for Training(category)
           n_classes (int): number of categories

       Returns
           X_train (numpy array): 4 dimension Training Sets(image)
           X_valid (numpy array): 4 dimension Validation Sets(image)
           y_train (numpy array): 1 dimension Training Sets(category)
           y_valid (numpy array): 1 dimension Validation Sets(category)
    """
    X_train = np.array([]); X_valid = np.array([])
    y_train = np.array([]); y_valid = np.array([])

    sum_of_each_categories = 0
    for nc in range(n_classes):
        sum_of_each_categories += np.sum(original_y_train == nc)
        x = original_X_train[sum_of_each_categories : sum_of_each_categories + sum_of_each_categories]
        y = original_y_train[sum_of_each_categories : sum_of_each_categories + sum_of_each_categories]
        train_feature, valid_feature, train_label, valid_label = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=3
        )
        if nc == 0:
            X_train = train_feature; X_valid = valid_feature
            y_train = train_label;   y_valid = valid_label
        else:
            X_train = np.concatenate([X_train, train_feature], axis=0)
            X_valid = np.concatenate([X_valid, valid_feature], axis=0)
            y_train = np.concatenate([y_train, train_label], axis=0)
            y_valid = np.concatenate([y_valid, valid_label], axis=0)

    return X_train, X_valid, y_train, y_valid

def to_onehot_vector(y_values, n_classes):
    """convert to one hot vector"""
    onehot_y = np.zeros((y_values.shape[0], n_classes))
    onehot_y[np.arange(y_values.shape[0]), y_values] = 1
    return onehot_y

def main():
    training_file = './train.p'
    test_file = './test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    with open(test_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']

    X_train = RGB_to_YUV(X_train)
    X_test = RGB_to_YUV(X_test)

    n_classes = len(set(y_train))

    X_train, X_valid, y_train, y_valid = divide_training_and_validataion(X_train, y_train, n_classes)

    X_train = X_train / 255
    X_valid = X_valid / 255
    X_test = X_test / 255

    X_train, y_train = shuffle_datasets(X_train, y_train)
    X_valid, y_valid = shuffle_datasets(X_valid, y_valid)

    Y_train = to_onehot_vector(y_train, n_classes)
    Y_valid = to_onehot_vector(y_valid, n_classes)
    Y_test = to_onehot_vector(y_test, n_classes)

    training_epochs = 200
    valid_accuracy_list, test_accuracy_list = train_validation_test(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, training_epochs=training_epochs, batch_size=378)

    plt.plot(np.arange(0,training_epochs), test_accuracy_list, 'b', label="test accuracy")
    plt.plot(np.arange(0,training_epochs), valid_accuracy_list, 'r', label="valid accuracy")
    plt.legend(loc='best')
    plt.yticks(np.arange(0.00, 1.05, 0.05))
    plt.xlabel("epoch");plt.ylabel("accuracy")
    plt.title("model.py"); plt.savefig("model.png", dpi=150)
    np.savez('model.npz', valid=valid_accuracy_list, test=test_accuracy_list)
    plt.show()


if __name__ == '__main__':
    main()
