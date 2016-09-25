import numpy as np
import tensorflow as tf
import pickle
import sys
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

n_labels = 10
n_channels = 1
image_width = 28
n_hidden = 256
n_input = image_width ** 2
bottleneck = 10

try:
    num_to_make = int(sys.argv[1])
    print('Number of foolers to generate:', num_to_make)
except:
    print('Defaulted to making one fooling image')
    num_to_make = 1

graph = tf.Graph()
with graph.as_default():
    # fixing input to be 1 because we know input size
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.int64, [None])

    W = {
        '1': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_input, n_hidden]), 0)),
        '2': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
        'out': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_labels]), 0)),
        'decodeh2': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, bottleneck]), 0)),
        'out_info': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_labels, bottleneck]), 0)),
        'd1': tf.Variable(tf.nn.l2_normalize(tf.random_normal([bottleneck, n_hidden]), 0)),
        'salvage': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_input]), 0)),
    }

    b = {
        '1': tf.Variable(tf.zeros([n_hidden])),
        '2': tf.Variable(tf.zeros([n_hidden])),
        'out': tf.Variable(tf.zeros([n_labels])),
        'd2': tf.Variable(tf.zeros([bottleneck])),
        'd1': tf.Variable(tf.zeros([n_hidden])),
        'salvage': tf.Variable(tf.zeros([n_input])),
    }

    def gelu_fast(__x):
        return 0.5 * __x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (__x + 0.044715 * tf.pow(__x, 3))))
    f = gelu_fast

    def model(_x):
        h1 = f(tf.matmul(_x, W['1']) + b['1'])
        h2 = f(tf.matmul(h1, W['2']) + b['2'])
        out = tf.matmul(h2, W['out']) + b['out']

        decode2 = f(tf.matmul(h2, W['decodeh2']) + tf.matmul(out, W['out_info']) + b['d2'])
        decode1 = f(tf.matmul(decode2, W['d1']) + b['d1'])
        salvaged = tf.matmul(decode1, W['salvage']) + b['salvage']

        return out, salvaged

    pred, recon = model(x)

    starter_learning_rate = 0.001
    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
    loss = ce + 1*tf.reduce_mean(tf.square(x - recon))
    lr = tf.constant(0.001)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    wrong_pred = tf.not_equal(tf.argmax(pred, 1), y)
    compute_error = 100. * tf.reduce_mean(tf.to_float(wrong_pred))

sess = tf.InteractiveSession(graph=graph)
tf.initialize_all_variables().run()

batch_size = 128
training_epochs = 20
num_batches = int(mnist.train.num_examples / batch_size)
ce_ema = 2.3            # - log(0.1)
err_ema = 0.9
risk_loss_ema = 0.3     # - log(0.5)
learning_rate = 0.001
for epoch in range(training_epochs):
    if epoch >= 20:
        learning_rate = 0.0001
    for i in range(num_batches):
        bx, by = mnist.train.next_batch(batch_size)
        _, err, l = sess.run([optimizer, compute_error, ce], feed_dict={x: bx, y: by, lr: learning_rate})
        ce_ema = ce_ema * 0.95 + 0.05 * l
        err_ema = err_ema * 0.95 + 0.05 * err
    for i in range(mnist.validation.num_examples // batch_size):
        bx, by = mnist.validation.next_batch(batch_size)
        _, err, l = sess.run([optimizer, compute_error, ce], feed_dict={x: bx, y: by, lr: learning_rate})
        ce_ema = ce_ema * 0.95 + 0.05 * l
        err_ema = err_ema * 0.95 + 0.05 * err

    print('Epoch number:', epoch, 'Error EMA:', err_ema, 'CE EMA', ce_ema)

print('Done training')

mean_img = tf.reshape(tf.reduce_mean(mnist.train.next_batch(1000)[0], 0, keep_dims=True), [28,28]).eval()

# def show_image(img, rescale=False, add_mean=False):
#     img = img.reshape(28,28)
#
#     img = img.copy()
#     if add_mean:
#         img += mean_img
#     if rescale:
#         low, high = np.min(img), np.max(img)
#         img = (img - low) / (high - low)
#     plt.imshow(img, vmin=0, vmax=1)
#     plt.gca().axis('off')

def make_fooling_image(image, target, reg=1., step=1/255., max_iters=1000, confidence_thresh=0.5):
    fooling_image = image.copy()

    for j in range(max_iters):
        dFool, decoded, probs = sess.run([tf.gradients(ce + reg*tf.reduce_sum(tf.square(image[0] -
                                                                                          fooling_image[0]))/2., x)[0],
                                          recon, tf.nn.softmax(model(fooling_image)[0])],
                                         feed_dict={x: fooling_image, y: [target]})
        fooling_image[0] -= step * (np.squeeze(dFool[0]))
        fooling_image[0] = np.clip(fooling_image[0], 0, 1)  # poor man's box constraints
        fool_prob = probs[0, target]

        if j % 10 == 0:
            print('Fooling Image Probability Percent (iter %s): %s' % (j, 100.*fool_prob))
        if fool_prob > confidence_thresh:
            print('Final food prob percent:', 100*fool_prob)
            break

    return fooling_image, decoded


l1_distances = []
l2_distances = []
linf_distances = []

try:
    history = pickle.load(open("./data/mnist_foolers_reconstruction.p", "rb"))
except:
    history = {}

if not os.path.exists('./data'):
    os.makedirs('./data')

for i in range(num_to_make):
    image, true_y = mnist.test.next_batch(1)

    # ensure the network gets our current example correct
    while true_y != tf.argmax(model(tf.constant(image))[0], 1).eval()[0]:
        image, true_y = mnist.test.next_batch(1)

    target_y = np.random.choice(10)
    while target_y == true_y:
        target_y = np.random.choice(10)

    fooling_image, decoded = make_fooling_image(image, target_y)
    confidence = sess.run(tf.nn.softmax(model(fooling_image)[0])[0, target_y])

    if confidence < 0.5:
        fooled = 'not_fooled'
        print('Network is NOT fooled!')
    else:
        fooled = 'fooled'
        print('Network is fooled!')

    if fooled == 'fooled':
        l2 = np.sum(np.square(image - fooling_image))
        l1 = np.sum(np.abs(image - fooling_image))
        linf = np.sum(np.max(np.abs(image - fooling_image)))

        l2_distances.append(l2)
        l1_distances.append(l1)
        linf_distances.append(linf)

        history[str(i)] = [true_y, target_y, fooling_image, image,
                           decoded, sess.run(model(tf.constant(image))[1]), l2, l1]

    print('Number of fooling examples collected:', len(l2_distances))
    print('L1 mean:', np.mean(np.array(l1_distances)))
    print('L2 mean:', np.mean(np.array(l2_distances)))
    print('LInf mean:', np.mean(np.array(linf_distances)))

    pickle.dump(history, open("./data/mnist_foolers_reconstruction.p", "wb"))
