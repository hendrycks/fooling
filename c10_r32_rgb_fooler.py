import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os
import pickle as pickle
from six.moves import urllib
import tarfile
import scipy.stats.mstats
from load_cifar10 import load_data

# training parameters
initial_learning_rate = 0.001
training_epochs = 200
batch_size = 128

# architecture parameters
n_labels = 10
crop_length = 32
n_channels = 3
image_width = 32
n_input = 32 * 32
mode = 'normal'
nonlinearity_name = 'relu'

try:
    num_to_make = int(sys.argv[1])
    print('Number of foolers to generate:', num_to_make)
except:
    print('Defaulted to making one fooling image')
    num_to_make = 1

# only normal and bl_normal make sense when YUV is unscaled; uncomment rescaling code to soundly use other options

try:
    mode = sys.argv[2]       # 'energy_blur', 'bl_energy', 'normal', 'bl', 'mix2'
    print('Chosen mode:', mode)
except:
    print('Defaulted to normal mode since no mode given through command line')
    mode = 'normal'

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

# try:
#     wiggle = str2bool(sys.argv[3])       # True, False
#     print('Wiggle image?', wiggle)
# except:
#     print('Defaulted to no image wiggling')
#     wiggle = False
wiggle = False

try:
    manifold_step = float(sys.argv[3])
    print('manifold step:', manifold_step)
except:
    print('Default to a manifold step of 0.02 (used in class recovery)')
    manifold_step = 0.02

try:
    recover = str2bool(sys.argv[4])       # True, False
    print('Recover class?', recover)
except:
    print('Defaulted to no class recovery')
    recover = False

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=[None, crop_length, crop_length, n_channels])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    is_training = tf.constant(False)     # tf.placeholder(tf.bool)

    W = {}
    bn = {}

    params = pickle.load(open("./data/r32_rgb.pkl", "rb"), encoding='latin1')

    for layer in range(0, 31):
        # awkward offset because of bn for input
        l_str = str(layer+1)
        W['filter' + l_str] = tf.Variable(np.moveaxis(params[layer * 5], [0, 1, 2, 3], [3, 2, 0, 1]))
        bn['beta' + l_str] = tf.Variable(params[layer * 5 + 1])
        bn['gamma' + l_str] = tf.Variable(params[layer * 5 + 2])
        bn['mu' + l_str] = tf.constant(params[layer * 5 + 3])
        bn['inv_std' + l_str] = tf.constant(params[layer * 5 + 4])

    W['w_out'] = tf.Variable(params[155])
    W['b_out'] = tf.Variable(params[156])


    def feedforward(_x, n=5):
        rho = tf.nn.relu

        def residual_block(h, layer_number=1, input_num_filters=32, increase_dim=False):
            l_num = str(layer_number)
            if increase_dim:
                first_stride = [1, 2, 2, 1]
                out_num_filters = input_num_filters * 2
            else:
                first_stride = [1, 1, 1, 1]
                out_num_filters = input_num_filters

            stack1 = rho((tf.nn.conv2d(h, W['filter' + l_num], strides=first_stride, padding='SAME') -
                          bn['mu' + l_num]) * bn['inv_std' + l_num] * bn['gamma' + l_num] + bn['beta' + l_num])

            l_num = str(layer_number + 1)
            stack2 = (tf.nn.conv2d(stack1, W['filter' + l_num], strides=[1, 1, 1, 1], padding='SAME') -
                      bn['mu' + l_num]) * bn['inv_std' + l_num] * bn['gamma' + l_num] + bn['beta' + l_num]

            if increase_dim:
                # upgrade tensorflow h[:, ::2, ::2, :]
                # array_ops.strided_slice(h, [0,0,0,0], [2000,-1,-1,input_num_filters], [1,2,2,1])
                h_squished = tf.nn.max_pool(h, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
                padded = tf.pad(h_squished, [[0, 0], [0, 0], [0, 0], [out_num_filters // 4, out_num_filters // 4]])
                block = rho(stack2 + padded)
            else:
                block = rho(stack2 + h)

            return block

        x_input = _x
        # bsize x 32 x 32 x 16
        l = rho((tf.nn.conv2d(x_input, W['filter1'], strides=[1, 1, 1, 1], padding='SAME') -
                 bn['mu1']) * bn['inv_std1'] * bn['gamma1'] + bn['beta1'])

        # bsize x 32 x 32 x 16
        for i in range(n):
            l = residual_block(l, layer_number=2 * i + 2)

        # bsize x 16 x 16 x 32
        l = residual_block(l, increase_dim=True, layer_number=2 * n + 2, input_num_filters=16)
        for i in range(1, n):
            l = residual_block(l, layer_number=2 * n + 2 * i + 2)

        # bsize x 8 x 8 x 64
        l = residual_block(l, increase_dim=True, layer_number=4 * n + 2, input_num_filters=32)
        for i in range(1, n):
            l = residual_block(l, layer_number=4 * n + 2 * i + 2)

        l = tf.reduce_mean(l, reduction_indices=[1, 2])

        return tf.matmul(l, W['w_out']) + W['b_out']


    def normal(_x):
        return feedforward(_x)


    def energy_blur(_x):
        _x = tf.reshape(_x, [-1, image_width, image_width, 3])
        # 5x5, sigma = 0.7
        filter = tf.reshape(tf.constant([[0.000252, 0.00352, 0.008344, 0.00352, 0.000252],
                                         [0.00352, 0.049081, 0.11634, 0.049081, 0.00352],
                                         [0.008344, 0.11634, 0.275768, 0.11634, 0.008344],
                                         [0.00352, 0.049081, 0.11634, 0.049081, 0.00352],
                                         [0.000252, 0.00352, 0.008344, 0.00352, 0.000252]],
                                        dtype=tf.float32), [5, 5, 1, 1])
        h, s, v = tf.split(3, 3, _x)
        # this filter is not for 32x32 but 28x28
        # h = tf.nn.conv2d(1. / tf.cos(h * (np.pi - 0.5) / 2.) - 1,
        #                  filter, strides=[1, 1, 1, 1], padding='SAME')
        # h = tf.sqrt(tf.reshape(h, [-1, 32, 32, 1]) + 1e-12)
        # s = tf.nn.conv2d(1. / tf.cos(s * (np.pi - 0.5) / 2.) - 1,
        #                  filter, strides=[1, 1, 1, 1], padding='SAME')
        # s = tf.sqrt(tf.reshape(s, [-1, 32, 32, 1]) + 1e-12)
        # v = tf.nn.conv2d(1. / tf.cos(v * (np.pi - 0.5) / 2.) - 1,
        #                  filter, strides=[1, 1, 1, 1], padding='SAME')
        # v = tf.sqrt(tf.reshape(v, [-1, 32, 32, 1]) + 1e-12)
        h = tf.nn.conv2d(tf.square(h), filter, strides=[1, 1, 1, 1], padding='SAME')
        h = tf.sqrt(tf.reshape(h, [-1, 32, 32, 1]) + 1e-12)
        s = tf.nn.conv2d(tf.square(s), filter, strides=[1, 1, 1, 1], padding='SAME')
        s = tf.sqrt(tf.reshape(s, [-1, 32, 32, 1]) + 1e-12)
        v = tf.nn.conv2d(tf.square(v), filter, strides=[1, 1, 1, 1], padding='SAME')
        v = tf.sqrt(tf.reshape(v, [-1, 32, 32, 1]) + 1e-12)

        _x = tf.concat(3, [h, s, v])
        return feedforward(_x)


    def bilateralFilterSpace2(image, sigma_range):
        sigma_space = 2
        win_size = max(5., 2 * np.ceil(3 * sigma_space) + 1)

        win_ext = int((win_size - 1) / 2)
        height = 32
        width = 32

        spatial_gaussian = []
        for i in range(-win_ext, win_ext + 1):
            for j in range(-win_ext, win_ext + 1):
                spatial_gaussian.append(np.exp(-0.5 * (i ** 2 + j ** 2) / sigma_space ** 2))

        # we use "symmetric" as it best approximates "edge" padding
        padded = tf.pad(image, [[win_ext, win_ext], [win_ext, win_ext]], mode='SYMMETRIC')
        out_image = tf.zeros(tf.shape(image))
        weight = tf.zeros(tf.shape(image))

        spatial_index = 0
        row = -win_ext
        col = -win_ext

        # first row
        # 1
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 2
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 3
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 4
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 5
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 6
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 7
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 8
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 9
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 10
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 11
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 12
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 13
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1

        row += 1
        col = -win_ext

        # row 2
        # 1
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 2
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 3
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 4
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 5
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 6
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 7
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 8
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 9
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 10
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 11
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 12
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 13
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1

        row += 1
        col = -win_ext

        # row 3
        # 1
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 2
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 3
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 4
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 5
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 6
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 7
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 8
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 9
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 10
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 11
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 12
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 13
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1

        row += 1
        col = -win_ext

        # row four
        # 1
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 2
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 3
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 4
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 5
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 6
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 7
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 8
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 9
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 10
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 11
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 12
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 13
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1

        row += 1
        col = -win_ext

        # row five
        # 1
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 2
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 3
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 4
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 5
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 6
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 7
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 8
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 9
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 10
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 11
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 12
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 13
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1

        row += 1
        col = -win_ext

        # row six
        # 1
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 2
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 3
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 4
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 5
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 6
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 7
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 8
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 9
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 10
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 11
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 12
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 13
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1

        row += 1
        col = -win_ext

        # row seven
        # 1
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 2
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 3
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 4
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 5
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 6
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 7
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 8
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 9
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 10
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 11
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 12
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 13
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1

        row += 1
        col = -win_ext

        # row eight
        # 1
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 2
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 3
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 4
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 5
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 6
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 7
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 8
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 9
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 10
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 11
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 12
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 13
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1

        row += 1
        col = -win_ext

        # row nine
        # 1
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 2
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 3
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 4
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 5
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 6
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 7
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 8
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 9
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 10
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 11
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 12
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 13
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1

        row += 1
        col = -win_ext

        # row 10
        # 1
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 2
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 3
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 4
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 5
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 6
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 7
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 8
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 9
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 10
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 11
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 12
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 13
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1

        row += 1
        col = -win_ext

        # row 11
        # 1
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 2
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 3
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 4
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 5
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 6
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 7
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 8
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 9
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 10
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 11
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 12
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 13
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1

        row += 1
        col = -win_ext

        # row 12
        # 1
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 2
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 3
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 4
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 5
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 6
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 7
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 8
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 9
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 10
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 11
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 12
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 13
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1

        row += 1
        col = -win_ext

        # row 13
        # 1
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 2
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 3
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 4
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 5
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 6
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 7
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 8
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 9
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 10
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 11
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 12
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1
        col += 1

        # 13
        sub_image = tf.slice(padded, [win_ext + row, win_ext + col], [height, width])
        value = tf.exp(-0.5 * (((image - sub_image) / sigma_range) ** 2)) * spatial_gaussian[spatial_index]
        out_image += value * sub_image
        weight += value
        spatial_index += 1

        row += 1
        col = -win_ext

        # done!
        out_image /= weight
        return out_image


    def bl_energy_preprocess(_x, srange=0.5):
        _x = tf.reshape(_x, [-1, image_width, image_width, 3])
        h, s, v = tf.split(3, 3, _x)
        # h = bilateralFilterSpace2(tf.reshape(1. / tf.cos(h * (np.pi - 0.5) / 2.) - 1, [32, 32]), srange)
        # h = tf.sqrt(tf.reshape(h, [-1, 32, 32, 1]) + 1e-12)
        # s = bilateralFilterSpace2(tf.reshape(1. / tf.cos(s * (np.pi - 0.5) / 2.) - 1, [32, 32]), srange)
        # s = tf.sqrt(tf.reshape(s, [-1, 32, 32, 1]) + 1e-12)
        # v = bilateralFilterSpace2(tf.reshape(1. / tf.cos(v * (np.pi - 0.5) / 2.) - 1, [32, 32]), srange)
        # v = tf.sqrt(tf.reshape(v, [-1, 32, 32, 1]) + 1e-12)
        h = bilateralFilterSpace2(tf.reshape(tf.square(h), [32,32]), srange)
        h = tf.sqrt(tf.reshape(h, [-1, 32, 32, 1])+1e-12)
        s = bilateralFilterSpace2(tf.reshape(tf.square(s), [32,32]), srange)
        s = tf.sqrt(tf.reshape(s, [-1, 32, 32, 1])+1e-12)
        v = bilateralFilterSpace2(tf.reshape(tf.square(v), [32,32]), srange)
        v = tf.sqrt(tf.reshape(v, [-1, 32, 32, 1])+1e-12)

        _x = tf.concat(3, [h, s, v])
        return _x


    def bl_energy(_x, srange=0.5):
        return feedforward(bl_energy_preprocess(_x, srange))


    def bl_normal(_x, srange=0.5):
        _x = tf.reshape(_x, [-1, image_width, image_width, 3])
        h, s, v = tf.split(3, 3, _x)
        h = tf.reshape(bilateralFilterSpace2(tf.reshape(h, [32, 32]), srange), [-1, 32, 32, 1])
        s = tf.reshape(bilateralFilterSpace2(tf.reshape(s, [32, 32]), srange), [-1, 32, 32, 1])
        v = tf.reshape(bilateralFilterSpace2(tf.reshape(v, [32, 32]), srange), [-1, 32, 32, 1])
        conv = tf.concat(3, [h, s, v])
        return feedforward(conv)


    if wiggle:
        h, s, v = tf.split(3, 3, x)

        low, high = tf.reduce_min(h), tf.reduce_max(h)
        new_low = tf.random_uniform(shape=[1], minval=-0.05, maxval=0.05) + low
        new_high = tf.random_uniform(shape=[1], minval=-0.05, maxval=0.05) + high
        h = (h - low) / (high - low + 1e-11) * (new_high - new_low) + new_low

        low, high = tf.reduce_min(s), tf.reduce_max(s)
        new_low = tf.random_uniform(shape=[1], minval=-0.05, maxval=0.05) + low
        new_high = tf.random_uniform(shape=[1], minval=-0.05, maxval=0.05) + high
        s = (s - low) / (high - low + 1e-11) * (new_high - new_low) + new_low

        low, high = tf.reduce_min(v), tf.reduce_max(v)
        new_low = tf.random_uniform(shape=[1], minval=-0.05, maxval=0.05) + low
        new_high = tf.random_uniform(shape=[1], minval=-0.05, maxval=0.05) + high
        v = (v - low) / (high - low + 1e-11) * (new_high - new_low) + new_low

        x = tf.concat(3, [h, s, v])

    pred_normal = normal(x)
    loss_normal = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_normal, y))

    pred_energy_blur = energy_blur(x)
    loss_energy_blur = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_energy_blur, y))

    pred_bl_energy = bl_energy(x)
    loss_bl_energy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_bl_energy, y))

    pred_bl_stack = (bl_energy(x) + bl_energy(tf.clip_by_value(bl_energy_preprocess(x, 1.), 0, 1), 2.)) / 2.
    loss_bl_stack = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_bl_stack, y))

    pred_bl_normal = bl_normal(x)
    loss_bl_normal = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_bl_normal, y))

    pred_bln_stack = (bl_normal(x) + bl_normal(bl_energy_preprocess(x, 1.), 2.)) / 2.
    loss_bln_stack = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_bln_stack, y))

    if mode == 'normal':
        pred = pred_normal
        loss = loss_normal
    elif mode == 'energy_blur':
        pred = pred_energy_blur
        loss = loss_energy_blur
    elif mode == 'bl_energy':
        pred = pred_energy_blur
        loss = loss_energy_blur
    elif mode == 'stack':
        pred = pred_bln_stack
        loss = loss_bln_stack
    elif mode == 'mix_with_stack':
        pred = 2 / 3 * pred_bl_stack + 1 / 3 * pred_energy_blur
        loss = 2 / 3 * loss_bl_stack + 1 / 3 * loss_energy_blur
    elif mode == 'bl_normal':
        pred = pred_bl_normal
        loss = loss_bl_normal
    elif mode == 'mix2':
        pred = (pred_normal + pred_energy_blur) / 2.
        loss = loss_normal + loss_energy_blur
    elif mode == 'mix_energies':
        pred = (pred_bl_energy + pred_energy_blur) / 2.
        loss = loss_bl_energy + loss_energy_blur
    elif mode == 'mix3':
        pred = (pred_normal + pred_energy_blur + pred_bl_energy) / 3.
        loss = loss_normal + loss_energy_blur + loss_bl_energy
    else:  # mode 'mix4'
        pred = (pred_normal + pred_energy_blur + pred_bl_energy + pred_bl_normal) / 4.
        loss = loss_normal + loss_energy_blur + loss_bl_energy + loss_bl_normal

    # fooling_gradient = tf.gradients(loss, x)[0]

    # global_step = tf.Variable(0, trainable=False)
    # loss_ema = tf.Variable(2.3, trainable=False)
    # lr = tf.train.exponential_decay(initial_learning_rate, global_step, 50*390, .1, staircase=True)
    # optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

sess = tf.InteractiveSession(graph=graph)
tf.initialize_all_variables().run()
# saver = tf.train.Saver()

# saver.restore(sess, "./data/cifar10_r32_yuv_rescaled.ckpt")

# def rescale(img):
#     low, high = np.min(img), np.max(img)
#     return (img - low) / (high - low)

yuv_from_rgb = np.array([[0.299, 0.587, 0.114],
                         [-0.14714119, -0.28886916, 0.43601035],
                         [0.61497538, -0.51496512, -0.10001026]])

train_dataset, train_labels, _, _ = load_data(randomize=False)
mean_img = tf.reshape(np.mean(train_dataset, 0), [32, 32, 3]).eval()

# train_dataset = np.dot(train_dataset, yuv_from_rgb.T.copy())
# train_dataset[:, :, :, 0] = rescale(train_dataset[:, :, :, 0])
# train_dataset[:, :, :, 1] = rescale(train_dataset[:, :, :, 1])
# train_dataset[:, :, :, 2] = rescale(train_dataset[:, :, :, 2])

train_dataset = train_dataset.astype(np.float32)

# pred = sess.run(pred, feed_dict={x: train_dataset[0:3000,:,:,:]})
# error = np.argmax(pred, 1) != np.argmax(train_labels[0:3000, :], 1)
# print(np.mean(error))


class_names = ['airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def show_image(image, rescale=False, add_mean=True):
    image = image.reshape(32,32,3)

    img = image.copy()
#     rescale(img)
#     img = np.clip(img, 0, 1)
    # img[:,:,1] = (img[:,:,1] - 0)/(1 - 0) * (2 * 0.436) - 0.436
    # img[:,:,2] = (img[:,:,2] - 0)/(1 - 0) * (2 * 0.615) - 0.615

    # img = np.dot(img, np.linalg.inv(yuv_from_rgb).T.copy())
    # if add_mean:
    #     img += mean_img
    # if rescale:
    #     low, high = np.min(img), np.max(img)
    #     img = (img - low) / (high - low)

    plt.imshow(img)
    plt.gca().axis('off')

def rescale(img):
    low, high = np.min(img), np.max(img)
    return (img - low) / (high - low)

def get_random_label():
    random_label = np.random.uniform(-1, 1, size=(1, 10))
    low, high = np.min(random_label), np.max(random_label)
    # random_label = (random_label - low) / (high - low)

    return random_label.astype(np.float32)

# def make_auxiliary_image(image, step=2/255., max_iters=3, target_label=-1):
#     if target_label == -1:
#         random_label = get_random_label()
#     else:
#         random_label = [[0.] * 10]
#         random_label[0][target_label] += 1.
#
#     temp_fooling_image = image.copy()
#
#     for i in range(max_iters):
#         #         dFool = sess.run(tf.gradients(loss_normal, x), feed_dict={x: temp_fooling_image, y: random_label})
#         dFool = sess.run(tf.gradients(loss, x), feed_dict={x: temp_fooling_image, y: random_label})
#         temp_fooling_image[0] -= step * tf.squeeze(dFool[0]).eval()
#         temp_fooling_image[0] = np.clip(temp_fooling_image[0], 1e-10, 1)
#
#     return temp_fooling_image

def move_on_manifold(img, fool_class, step=manifold_step, max_iters=20, target_label=-1):
    # max iters should increase the harder the security layer is
    # and step should decrease with easier layers

    if target_label == -2:
        random_label = np.ones((1, 10)) / 10.
        random_label[0, target_y] -= 0.1
    elif target_label == -1:
        random_label = get_random_label()
    elif target_label != -1:
        random_label = [[0.] * 10]
        random_label[0][target_label] += 1.

    temp_fooling_image = img.copy()

    for i in range(max_iters):
        dFool, preds = sess.run([tf.gradients(loss, x)[0], pred],
                                feed_dict={x: temp_fooling_image, y: random_label})

        current_classification = np.argmax(preds)
        if current_classification != fool_class:
            return current_classification

        temp_fooling_image[0] -= step * tf.squeeze(dFool[0]).eval()
        # temp_fooling_image[0] = np.clip(temp_fooling_image[0], 0, 1)

    return fool_class

def recover_class(img, target_y):
    image_class = target_y
    other_classes = list(range(0, image_class)) + list(range(image_class + 1, 10))

    found_classes = []
    for c in other_classes:
        found_classes.append(move_on_manifold(img, image_class, target_label=c))
        print('Class guess:', found_classes[-1])

    # get random votes
    found_classes.append(move_on_manifold(img, image_class, target_label=-2))

    print(target_y, true_y, found_classes)
    class_mode = int(scipy.stats.mstats.mode(found_classes, axis=None)[0][0])

    if class_mode == image_class:
        result = -1     # -1 indicates the image was not a fooling image
    else:
        result = class_mode  # a nonzero class means the image was a fooler, and we return the right class

    return result


def make_fooling_image(image, target, reg=1e-4, step=5e-3, max_iters=50, confidence_thresh=0.5):
    orig_image = image.copy()
    fooling_image = image.copy()

    for _ in range(max_iters):
        dFool, predictions = sess.run([tf.gradients(loss, x)[0], pred], feed_dict={x: fooling_image, y: target})
        fooling_image[0] -= step * (tf.squeeze(dFool[0]).eval() + reg * (fooling_image[0] - orig_image[0]))
        # fooling_image[0] = np.clip(fooling_image[0], 0, 1)

        fool_prob = sess.run(tf.nn.softmax(predictions)[0, target_y])

        if fool_prob > confidence_thresh:
            break

    return fooling_image


l1_distances = []
l2_distances = []

# examples = [i for i in range(300, 400)]
# labels = [i % 10 for i in range(300, 400)]

try:
    history = pickle.load(open("./data/" + mode + "_r32_rgb.p", "rb"))
except:
    history = {}

for i in range(num_to_make):
    # choose source image from which to generate a fooling image
    rand_int = np.random.randint(50000, size=1)[0]
    # rand_int = examples[i]
    image, true_y = train_dataset[rand_int:rand_int+1], train_labels[rand_int]

    # ensure the network gets our current example correct
    while True:
        p = sess.run(pred, feed_dict={x: image})
        # it's not interesting to make a fooling image when the net doesn't even understand the source image
        if (tf.nn.softmax(p).eval())[0, true_y] > 0.75:
            break
        rand_int = np.random.randint(50000, size=1)[0]
        image, true_y = train_dataset[rand_int:rand_int+1], train_labels[rand_int]

    target_y = np.random.choice(list(range(0, true_y)) + list(range(min(9, true_y+1), 10)))
    # target_y = 2
    # target_y = labels[i]
    one_hot = [[0.]*10]
    one_hot[0][target_y] += 1.

    print('Rand int:', rand_int)

    fooling_image = make_fooling_image(image, one_hot)
    p = sess.run(pred, feed_dict={x: fooling_image})
    confidence = sess.run(tf.nn.softmax(p)[0, target_y])

    if confidence < 0.5:
        fooled = 'not_fooled'
        print('Network is NOT fooled!')
    else:
        fooled = 'fooled'
        print('Network is fooled!')

    # we must rescale since we subtracted the mean

    plt.figure(figsize=(3,1.8))
    plt.subplot(121)
    plt.title('Orig (%s)' % class_names[true_y])
    show_image(image, rescale=True, add_mean=False)

    plt.subplot(122)
    plt.title('Fool\n(%s)' % class_names[target_y])

    show_image(fooling_image, rescale=True, add_mean=False)
    plt.savefig('./data/rgb/' + mode + '/' + str(rand_int) + '_' + fooled + '.png')
    plt.close()

    if fooled == 'fooled':
        l2 = tf.reduce_sum(tf.square(image - fooling_image)).eval()
        l1 = tf.abs(image - fooling_image).eval()

        l2_distances.append(l2)
        l1_distances.append(l1)

        history[str(rand_int)] = [fooled, true_y, target_y, fooling_image, image, l2, l1]

    if recover and fooled == 'fooled':
        print('Recovering class')
        out_class = np.min(recover_class(fooling_image, target_y))  # in the event it returns more than one class
        print(out_class)
        if out_class == -1:
            print('Class not recovered')
        else:
            if out_class == true_y:
                print('Class successfully recovered')
            else:
                print('Class not recovered')

    print('Number of examples collected:', len(l2_distances))
    print('L1 mean:', np.mean(np.array(l1_distances)))
    print('L2 mean:', np.mean(np.array(l2_distances)))

    pickle.dump(history, open("./data/" + mode + "_r32_rgb.p", "wb"))
