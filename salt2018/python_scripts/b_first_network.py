import tensorflow as tf
import numpy as np
import sys, os,cv2
from sklearn.utils import shuffle
from scipy.misc import imread,imresize
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from skimage.transform import resize
from imgaug import augmenters as iaa
import nibabel as nib
import imgaug as ia
from scipy.ndimage import zoom
from sklearn.utils import shuffle
import matplotlib.animation as animation

# import layers
import sys
sys.path.append("..")
from a_all_layers import *

# load data
train_image = np.load('train_image.npy')
train_masks = np.load('train_masks.npy')
print(train_image.shape)
print(train_masks.shape)

# hyper
num_epoch = 100
learning_rate = 0.0008

beta1,beta2,adam_e = 0.9,0.999,1e-8

# class

# graph

# session








# -- end code --
