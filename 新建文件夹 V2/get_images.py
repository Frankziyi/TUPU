import numpy as np
import os
# from skimage import io
from scipy.misc import imread  # (h, w, c)
from scipy.misc import imresize

def get_images(ImageNameList, filepath):
	images = []
	for i in range(len(ImageNameList)):
		# Just for Inception_v3
		img_tmp = imread(filepath + ImageNameList[i])
		images.append(imresize(img_tmp, (299, 299)))
	images = np.array(images, dtype='f')
	return images