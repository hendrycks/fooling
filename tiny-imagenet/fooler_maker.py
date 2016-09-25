import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import cPickle as pickle

try:
    num_to_make = int(sys.argv[1])
    print 'Number of foolers to generate:', num_to_make
except:
    print 'Defaulted to making one fooling image'
    num_to_make = 1

try:
    mode = sys.argv[2]
    if mode == 'normal' or mode == 'fast' or mode == 'normal_clip' or mode == 'fast_clip':
        print 'Mode:', mode
    else:assert False, "Need mode to be 'normal', 'fast', 'normal_clip', or 'fast_clip'"
except:
    mode = 'normal'

# Load the TinyImageNet-100-A dataset and a pretrained model

from cs231n.data_utils import load_tiny_imagenet, load_models

tiny_imagenet_a = 'cs231n/datasets/tiny-imagenet-100-A'

class_names, X_train, y_train, X_val, y_val, X_test, y_test = load_tiny_imagenet(tiny_imagenet_a)

# Zero-mean the data
mean_img = np.mean(X_train, axis=0)
X_train -= mean_img
X_val -= mean_img
X_test -= mean_img

# Load a pretrained model; it is a five layer convnet.
models_dir = 'cs231n/datasets/tiny-100-A-pretrained'
model = load_models(models_dir)['model1']

from cs231n.classifiers.convnet import five_layer_convnet

# Array of shape (X_val.shape[0],) storing predictions on the validation set.
# y_val_pred[i] = c indicates that the model predicts that X_val[i] has label c.
y_val_pred = None


num_batches = 10
N_val = X_val.shape[0]
N_batches = N_val / num_batches
X_val_batches = np.array_split(X_val, num_batches)
y_val_batches = np.array_split(y_val, num_batches)

p = np.zeros((N_val, 100))
for i in xrange(num_batches):
    probs = five_layer_convnet(X_val_batches[i], model, return_probs=True)
    p[i * N_batches: (i + 1) * N_batches] = probs
y_val_pred = np.argmax(p, axis=1)

correct_indices, = np.nonzero(y_val_pred == y_val)

def show_image(img, rescale=False, add_mean=True):
    """
    Utility to show an image. In our ConvNets, images are 3D slices of 4D
    volumes; to visualize them we need to squeeze out the extra dimension,
    flip the axes so that channels are last, add the mean image, convert to
    uint8, and possibly rescale to be between 0 and 255. To make figures
    prettier we also need to suppress the axis labels after imshow.

    Input:
    - img: (1, C, H, W) or (C, H, W) or (1, H, W) or (H, W) giving
      pixel data for an image.
    - rescale: If true rescale the data to fit between 0 and 255
    - add_mean: If true add the training data mean image
    """
    img = img.copy()
    if add_mean:
        img += mean_img
    img = img.squeeze()
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    if rescale:
        low, high = np.min(img), np.max(img)
        img = 255.0 * (img - low) / (high - low)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')

def make_fooling_image(img, y, model, reg=0.0, step_size=1, confidence=0.5):
    """
    Perform optimization in image space to create an image that is similar to img
    but is classified as y by model.

    Inputs:
    - img: Array of shape (1, C, H, W) containing (mean-subtracted) pixel data for
      the starting point for the fooling image.
    - y: The desired label; should be a single integer.
    - model: Dictionary mapping parameter names to weights; this is a pretrained
      five_layer_net model.
    - reg: Regularization strength (in image space) for the fooling image. This
      is the parameter lambda in the equation above.
    - step_size: The step size to use for gradient descent.
    - confidence: The desired confidence threshold for the fooling image.
    """
    fooling_img = img.copy()

    i = 0
    while i < 100000:
        dX = five_layer_convnet(fooling_img, model, np.array([y]), compute_dX=True)
        dX += reg * (fooling_img - img)
        fooling_img -= step_size * dX

        if mode == 'normal_clip' or mode == 'fast_clip':
            fooling_img = np.clip(fooling_img, -np.min(mean_img), 255.-np.max(mean_img))

        prob = five_layer_convnet(fooling_img, model, return_probs=True)[0][y]
        if i % 1000 == 0: print prob
        i += 1
        if prob >= confidence:
            break

    return fooling_img

def make_fooling_image_fast(img, y, model, reg=0.0, step_size=10, confidence=0.5):
    # use fast gradient sign rather than an iterative procedure

    fooling_img = img.copy()

    dX = five_layer_convnet(fooling_img, model, np.array([y]), compute_dX=True)
    dX += reg * (fooling_img - img)
    fooling_img -= step_size * np.sign(dX)
    if mode == 'normal_clip' or mode == 'fast_clip':
        fooling_img = np.clip(fooling_img, -np.min(mean_img), 255.-np.max(mean_img))

    return fooling_img


# check if superfluous given the next command
if not os.path.exists("./data/"):
    os.makedirs("./data/")

if not os.path.exists("./data/tiny_foolers/"):
    os.makedirs("./data/tiny_foolers/")

if not os.path.exists("./data/tiny_foolers_fast/"):
    os.makedirs("./data/tiny_foolers_fast/")

if mode == 'normal':
    try:
        history = pickle.load(open("./data/tiny_foolers/tiny_foolers.p", "rb"))
    except:
        history = {}
elif mode == 'fast':
    try:
        history = pickle.load(open("./data/tiny_foolers_fast/tiny_foolers_fast.p", "rb"))
    except:
        history = {}
elif mode == 'fast_clip':
    try:
        history = pickle.load(open("./data/tiny_foolers_fast/tiny_foolers_fast_clip.p", "rb"))
    except:
        history = {}
elif mode == 'normal_clip':
    try:
        history = pickle.load(open("./data/tiny_foolers/tiny_foolers_clip.p", "rb"))
    except:
        history = {}


l1_distances, l2_distances = [], []

for i in range(num_to_make):
    idx = np.random.choice(np.nonzero(y_val_pred == y_val)[0])
    img = X_val[idx:idx+1]

    class_idx = np.random.randint(100)
    while class_idx == y_val[idx:idx+1][0]:
        class_idx = np.random.randint(100)

    # a step size of one takes forever
    if mode == 'normal' or mode == 'normal_clip':
        fooling_img = make_fooling_image(img, class_idx, model, step_size=1, reg=1e-5, confidence=0.5)
    elif mode == 'fast' or mode == 'fast_clip':
        # this has a high failure rate with step size 10 but we can't increase it lest people wrongly think
        # that it's too large a step size
        fooling_img = make_fooling_image_fast(img, class_idx, model, step_size=10, reg=1e-5, confidence=0.5)

    if five_layer_convnet(fooling_img, model, return_probs=True)[0, class_idx] >= 0.5:
        fooled = 'fooled'
    else: fooled = 'not_fooled'

    plt.figure(figsize=(10, 8.0))
    plt.subplot(121)
    plt.title('Orig.\n(%s)' % class_names[y_val[idx]][0])
    show_image(img, rescale=False, add_mean=True)

    plt.subplot(122)
    plt.title('Fooling\n(%s)' % class_names[class_idx][0])

    show_image(fooling_img, rescale=False, add_mean=True)
    if mode == 'normal':
        plt.savefig('./data/tiny_foolers/' + str(idx) + '_' + fooled + '.png')
    elif mode == 'fast':
        plt.savefig('./data/tiny_foolers_fast/' + str(idx) + '_' + fooled + '.png')
    # I don't care to render adversarial images that are clipped
    plt.close()

    if five_layer_convnet(fooling_img, model, return_probs=True)[0, class_idx] >= 0.5:
        l2 = np.sum(np.square(img - fooling_img))
        l1 = np.sum(np.abs(img - fooling_img))

        l2_distances.append(l2)
        l1_distances.append(l1)

        history[str(idx)] = [True, y_val[idx:idx+1][0], class_idx, fooling_img, img, l2, l1]

    print 'Number of examples collected:', len(l2_distances)
    print 'L1 mean:', np.mean(np.array(l1_distances))
    print 'L2 mean:', np.mean(np.array(l2_distances))

    if mode == 'normal':
        pickle.dump(history, open("./data/tiny_foolers/tiny_foolers.p", "wb"))
    elif mode == 'fast':
        pickle.dump(history, open("./data/tiny_foolers_fast/tiny_foolers_fast.p", "wb"))
    elif mode == 'fast_clip':
        pickle.dump(history, open("./data/tiny_foolers_fast/tiny_foolers_fast_clip.p", "wb"))
    elif mode == 'normal_clip':
        pickle.dump(history, open("./data/tiny_foolers/tiny_foolers_clip.p", "wb"))
