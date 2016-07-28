# Visible Progress on Adversarial Images and a New Saliency Map
This software allows users to reproduce the results in Visible Progress on Adversarial Images and a New Saliency Map, Dan Hendrycks and Kevin Gimpel 2016.

c10_r32_yuv_fooler_unscaled.py can generate yuv fooling images
PlotFoolers.ipynb makes visualizing the adversarial images easy
r32_rgb.pkl are the parameters for a Resnet trained on RGB images
r32_yuv_unscaled.pkl are the parameters for a Resnet trained on YUV images
resnet_32_yuv_unscaled.py generated the parameters for r32_yuv_unscaled.pkl
SaliencyMap.ipynb lets you play with the saliency map

# Execution
Please install Tensorflow, Lasagne, and Python 3+.
