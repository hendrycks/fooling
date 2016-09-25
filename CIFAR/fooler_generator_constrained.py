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
from load_cifar10 import load_data10

# architecture parameters
n_labels = 10
crop_length = 32
n_channels = 3
image_width = 32
n_input = 32 * 32

try:
    num_to_make = int(sys.argv[1])
    print('Number of foolers to generate:', num_to_make)
except:
    print('Defaulted to making one fooling image')
    num_to_make = 1

try:
    mode = sys.argv[2]       # 'kl', 'freq'
    print('Chosen mode:', mode)
    assert mode == 'kl' or mode == 'freq', 'Mode must be "kl" or "freq"'
except:
    print('Defaulted to kl mode since no mode given through command line')
    mode = 'kl'

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=[None, crop_length, crop_length, n_channels])
    y = tf.placeholder(dtype=tf.int64, shape=[None])
    is_training = tf.constant(False)     # tf.placeholder(tf.bool)

    W = {}
    bn = {}

    params = pickle.load(open("./r32.pkl", "rb"), encoding='latin1')

    bn['beta0'] = tf.Variable(params[0])
    bn['gamma0'] = tf.Variable(params[1])
    bn['mu0'] = tf.constant(params[2])
    bn['inv_std0'] = tf.constant(params[3])

    for layer in range(1, 32):
        # awkward offset because of bn for input
        l_str = str(layer)
        W['filter' + l_str] = tf.Variable(np.moveaxis(params[layer * 5 - 1], [0, 1, 2, 3], [3, 2, 0, 1]))
        bn['beta' + l_str] = tf.Variable(params[layer * 5 + 0])
        bn['gamma' + l_str] = tf.Variable(params[layer * 5 + 1])
        bn['mu' + l_str] = tf.constant(params[layer * 5 + 2])
        bn['inv_std' + l_str] = tf.constant(params[layer * 5 + 3])

    W['w_out'] = tf.Variable(params[159])
    W['b_out'] = tf.Variable(params[160])


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

        x_input = (_x - bn['mu0']) * bn['inv_std0'] * bn['gamma0'] + bn['beta0']
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

    pred = feedforward(x)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))

    s = tf.nn.softmax(pred)
    uniform_all = tf.ones(shape=tf.shape(s))/tf.to_float(tf.shape(s)[1])

    kl_all = tf.reduce_sum(s * tf.log(tf.abs(s) + 1e-11), reduction_indices=[1], keep_dims=True) \
             - tf.reduce_sum(s * tf.log(uniform_all), reduction_indices=[1], keep_dims=True)

    m_all, v_all = tf.nn.moments(kl_all, axes=[0])

    if mode == 'kl':
        barrier_center = tf.placeholder(tf.float32)
        barrier_radius = tf.placeholder(tf.float32)
        loss += tf.clip_by_value(-tf.log(barrier_center + barrier_radius - kl_all), 0, 1000)
        loss += tf.clip_by_value(-tf.log(kl_all - barrier_center + barrier_radius), 0, 1000)

    if mode == 'freq':
        freq_mean_low = tf.placeholder(tf.float32)
        freq_barrier_center_low = tf.placeholder(tf.float32)
        freq_barrier_radius_low = tf.placeholder(tf.float32)
        freq_mean_high = tf.placeholder(tf.float32)
        freq_barrier_center_high = tf.placeholder(tf.float32)
        freq_barrier_radius_high = tf.placeholder(tf.float32)

        loss += tf.clip_by_value(-tf.log(freq_barrier_center_low + freq_barrier_radius_low - freq_mean_low), 0, 1000)
        loss += tf.clip_by_value(-tf.log(freq_mean_low - freq_barrier_center_low + freq_barrier_radius_low), 0, 1000)
        loss += tf.clip_by_value(-tf.log(freq_barrier_center_high + freq_barrier_radius_high - freq_mean_high), 0, 1000)
        loss += tf.clip_by_value(-tf.log(freq_mean_high - freq_barrier_center_high + freq_barrier_radius_high), 0, 1000)

sess = tf.InteractiveSession(graph=graph)
tf.initialize_all_variables().run()

train_dataset, train_labels, test_dataset, test_labels = load_data10(randomize=False)
mean_img = np.reshape(np.mean(train_dataset, 0), (32, 32, 3))

train_dataset = train_dataset.astype(np.float32)
test_dataset = test_dataset.astype(np.float32)

# pred = sess.run(pred, feed_dict={x: train_dataset[0:3000,:,:,:]})
# error = np.argmax(pred, 1) != np.argmax(train_labels[0:3000, :], 1)
# print(np.mean(error))

# build up KL divergence statistics
predictions, kl_a = sess.run([pred, kl_all], feed_dict={x: test_dataset})
masked = sess.run(tf.boolean_mask(tf.constant(kl_a), tf.equal(tf.argmax(tf.constant(predictions), 1),
                                                              tf.to_int64(tf.constant(test_labels)))))
b_center = np.mean(masked)
b_radius = np.std(masked)/4.

# build up typical frequency statistics
cov = np.dot((train_dataset.reshape((-1, 32*32*3)) - mean_img.reshape((-1, 32*32*3))).T,
             (train_dataset.reshape((-1, 32*32*3)) - mean_img.reshape((-1, 32*32*3))))/train_dataset.shape[0]
u,s,v = np.linalg.svd(cov)

# note we're using PCA whitening, not ZCA
X_whitened = np.dot(train_dataset.reshape((-1, 32*32*3)), u)/np.sqrt(s + 1e-11)

# we choose 250 because it's the elbow of the variance captured
stds_low_freq = np.std(X_whitened[:,1500:2500], axis=1)
low_freq_mean = np.mean(stds_low_freq)
low_freq_std = np.std(stds_low_freq)

stds_high_freq = np.std(X_whitened[:,2500:], axis=1)
high_freq_mean = np.mean(stds_high_freq)
high_freq_std = np.std(stds_high_freq)

def whiten(unwhitened):
    img = unwhitened.copy()
    return np.dot(img.reshape((-1, 32*32*3)), u)/np.sqrt(s + 1e-11)

def get_freq_stats(unwhitened):
    whitened = whiten(unwhitened)

    # this should be just over one example
    stds_low_freq_example = np.std(whitened[:,1500:2500], axis=1)
    low_freq_mean_example = np.mean(stds_low_freq_example)
    stds_high_freq_example = np.std(whitened[:,2500:], axis=1)
    high_freq_mean_example = np.mean(stds_high_freq_example)

    return low_freq_mean_example, high_freq_mean_example


class_names = ['airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def show_image(image, rescale=False, add_mean=False):
    img = image.copy()
    img = img.reshape(32,32,3)

    # if add_mean:
    #     img += mean_img
    # if rescale:
    #     low, high = np.min(img), np.max(img)
    #     img = (img - low) / (high - low)

    plt.imshow(img)
    plt.gca().axis('off')

def make_fooling_image(image, target, reg=1e-3, step=1/255., max_iters=1000, confidence_thresh=0.5):
    # NOTE: we clip as a consequence of our discussion about improperly plotted images

    orig_image = image.copy()   # paranoia
    fooling_image = image.copy()

    for i in range(max_iters):
        if mode == 'kl':
            dFool, predictions = sess.run([tf.gradients(loss, x)[0], pred],
                                          feed_dict={x: fooling_image, y: [target],
                                                     barrier_center: b_center,
                                                     barrier_radius: b_radius})
        elif mode == 'freq':
            low, high = get_freq_stats(fooling_image)
            # print(low, high)
            dFool, predictions = sess.run([tf.gradients(loss, x)[0], pred],
                                          feed_dict={x: fooling_image, y: [target],
                                                     freq_mean_low: low,
                                                     freq_barrier_center_low: low_freq_mean,
                                                     freq_barrier_radius_low: low_freq_std,
                                                     freq_mean_high: high,
                                                     freq_barrier_center_high: high_freq_mean,
                                                     freq_barrier_radius_high: high_freq_std,
                                                     })

        fooling_image[0] -= step * (np.squeeze(dFool[0]) + reg * (fooling_image[0] - orig_image[0]))
        fooling_image[0] = np.clip(fooling_image[0], 0, 1)

        if mode == 'freq':
            low, high = get_freq_stats(fooling_image)
            if np.abs(low - low_freq_mean) > low_freq_std or np.abs(high - high_freq_mean) > high_freq_std:
                return orig_image   # optimization failed

        if mode == 'kl':
            if np.abs(sess.run(kl_all, feed_dict={x: image}) - b_center) > b_radius:
                return orig_image       # optimization failed

        fool_prob = sess.run(tf.nn.softmax(predictions)[0, target])

        if i % 100 == 0:
            print('Current fool prob (not a percent)', fool_prob, 'Step:', i)
        if fool_prob > confidence_thresh:
            print('Exiting with this fool prob(not a percent)', fool_prob, 'at step', i)
            break

    return fooling_image

l1_distances = []
l2_distances = []
linf_distances = []

# examples = [i for i in range(300, 400)]
# labels = [i % 10 for i in range(300, 400)]

try:
    history = pickle.load(open("./data/" + mode + "_foolers.p", "rb"))
except:
    history = {}

if not os.path.exists('./data'):
    os.makedirs('./data')

if not os.path.exists('./data/normal'):
    os.makedirs('./data/normal')

if not os.path.exists('./data/mix'):
    os.makedirs('./data/mix')

if not os.path.exists('./data/freq'):
    os.makedirs('./data/freq')

if not os.path.exists('./data/kl'):
    os.makedirs('./data/kl')

for i in range(num_to_make):
    # choose source image from which to generate a fooling image
    rand_int = np.random.randint(10000, size=1)[0]
    # rand_int = examples[i]
    image, true_y = test_dataset[rand_int:rand_int+1], test_labels[rand_int]

    # ensure the network gets our current example correct
    while True:
        p = sess.run(pred, feed_dict={x: image})
        # it's not interesting to do a source-target attack when the net doesn't even understand the source image
        if mode == 'kl':
            if (tf.nn.softmax(p).eval())[0, true_y] > 0.5 and np.abs(sess.run(kl_all, feed_dict={x: image}) - b_center) < b_radius/2:
                break
        elif mode == 'freq':
            low, high = get_freq_stats(image)
            if (tf.nn.softmax(p).eval())[0, true_y] > 0.5 and np.abs(low - low_freq_mean) < 2*low_freq_std/3 and np.abs(high - high_freq_mean) < 2*high_freq_std/3:
                break
        rand_int = np.random.randint(10000, size=1)[0]
        image, true_y = test_dataset[rand_int:rand_int+1], test_labels[rand_int]

    target_y = np.random.choice(10)
    while target_y == true_y:
        target_y = np.random.choice(10)

    print('Rand int:', rand_int)

    fooling_image = make_fooling_image(image, target_y)
    p = sess.run(pred, feed_dict={x: fooling_image})
    confidence = sess.run(tf.nn.softmax(p)[0, target_y])

    if confidence < 0.5:
        fooled = 'not_fooled'
        print('Network is NOT fooled!')
    else:
        fooled = 'fooled'
        print('Network is fooled!')

    plt.figure(figsize=(3, 1.8))
    plt.subplot(121)
    plt.title('Orig (%s)' % class_names[true_y])
    show_image(image, rescale=False, add_mean=False)

    plt.subplot(122)
    plt.title('Fool\n(%s)' % class_names[target_y])

    show_image(fooling_image, rescale=False, add_mean=False)
    plt.savefig('./data/' + mode + '/' + str(rand_int) + '_' + fooled + '.png')
    plt.close()

    if fooled == 'fooled':
        l2 = np.sum(np.square(image - fooling_image))
        l1 = np.sum(np.abs(image - fooling_image))
        linf = np.sum(np.max(np.abs(image - fooling_image)))

        l2_distances.append(l2)
        l1_distances.append(l1)
        linf_distances.append(linf)

        history[str(rand_int)] = [fooled, true_y, target_y, fooling_image, image, l2, l1]

    print('Number of fooling examples collected:', len(l2_distances))
    print('L1 mean:', np.mean(np.array(l1_distances)))
    print('L2 mean:', np.mean(np.array(l2_distances)))
    print('LInf mean:', np.mean(np.array(linf_distances)))

    pickle.dump(history, open("./data/" + mode + "_foolers.p", "wb"))
