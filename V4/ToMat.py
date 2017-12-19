from scipy import io
import numpy as np

mat1 = np.load('test_probe_features.npy')
io.savemat('test_probe_features.mat', {'test_probe_features': mat1})

mat2 = np.load('test_probe_labels.npy')
io.savemat('test_probe_labels.mat', {'test_probe_labels': mat2})

mat3 = np.load('test_gallery_features.npy')
io.savemat('test_gallery_features.mat', {'test_gallery_features': mat3})

mat4 = np.load('test_gallery_labels.npy')
io.savemat('test_gallery_labels.mat', {'test_gallery_labels': mat4})