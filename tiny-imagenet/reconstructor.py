import numpy as np
import tensorflow as tf

n_input = 64*64*3
n_hidden = 1024

graph = tf.Graph()
with graph.as_default():
    # fixing input to be 1 because we know input size
    x = tf.placeholder(tf.float32, [None, n_input])
    W = {
        '1': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_input, n_hidden]), 0)),
        '2': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
        '3': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
        '4': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
        '5': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_hidden]), 0)),
        'salvage': tf.Variable(tf.nn.l2_normalize(tf.random_normal([n_hidden, n_input]), 0)),
    }

    b = {
        '1': tf.Variable(tf.zeros([n_hidden])),
        '2': tf.Variable(tf.zeros([n_hidden])),
        '3': tf.Variable(tf.zeros([n_hidden])),
        '4': tf.Variable(tf.zeros([n_hidden])),
        '5': tf.Variable(tf.zeros([n_hidden])),
        'salvage': tf.Variable(tf.zeros([n_input])),
    }

    def gelu_fast(__x):
        return 0.5 * __x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (__x + 0.044715 * tf.pow(__x, 3))))
    f = gelu_fast

    def model(_x):
        h1 = f(tf.matmul(_x, W['1']) + b['1'])
        h2 = f(tf.matmul(h1, W['2']) + b['2'])
        h3 = f(tf.matmul(h2, W['3']) + b['3'])
        h4 = f(tf.matmul(h3, W['4']) + b['4'])
        h5 = f(tf.matmul(h4, W['5']) + b['5'])

        return tf.matmul(h5, W['salvage']) + b['salvage']

    recon = model(x)

    loss = tf.reduce_mean(tf.square(x - recon))
    lr = tf.constant(0.001)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

u = np.load('u.npy')
s = np.load('s.npy')
v = np.load('v.npy')
mean_img = np.load('mean.npy')

from cs231n.data_utils import load_tiny_imagenet

tiny_imagenet_a = 'cs231n/datasets/tiny-imagenet-100-A'

class_names, X_train, y_train, X_val, y_val, X_test, y_test = load_tiny_imagenet(tiny_imagenet_a)

print X_train.shape
X_train = np.vstack((X_train, X_val)).transpose(0, 2, 3, 1).reshape((-1, 64*64*3))
print X_train.shape

X_train = np.dot(X_train, u)
X_train = X_train/np.sqrt(s + 1e-11)

sess = tf.InteractiveSession(graph=graph)
tf.initialize_all_variables().run()
saver = tf.train.Saver(max_to_keep=1)

learning_rate = 0.001
batch_size = 64
num_batches = int(X_train.shape[0] / batch_size)
l_ema = 10
print('Beginning training')
for epoch in range(30):
    if epoch >= 20:
        learning_rate = 0.0001
    for i in range(num_batches):
        offset = batch_size*i
        _, l = sess.run([optimizer, loss], feed_dict={x: X_train[offset:offset+batch_size,:], lr: learning_rate})
        l_ema = l_ema * 0.95 + 0.05 * l

    print('Epoch number:', epoch, 'Loss EMA', l_ema)
    saver.save(sess, './reconstructor.ckpt')
