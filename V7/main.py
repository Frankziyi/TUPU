import tensorflow as tf
# from network_def import inception_reid
from network_def import inception_v3
from network_def import utils
import numpy as np
import os
import MyUtils

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

dataset_path = '../../world/data-gpu-94/sysu-reid/person-reid-data/Market-1501/Market-1501-v15.09.15/'
trainset_path = dataset_path + '/bounding_box_train/' # 751 id
gallery_path = dataset_path + '/bounding_box_test/' # 750 id
probe_path = dataset_path + '/query/'

batch_size = 32
num_classes = 1501
image_height = 225
image_width = 225
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999

# Train_Images = tf.placeholder(tf.float32, shape=(batch_size, 128, 64, 3))
Train_Images = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, 3))
Train_Labels = tf.placeholder(tf.int32, shape=batch_size)

# logits, end_points = inception_reid.inception_reid(Train_Images, num_classes, asoftmax=False)

logits, end_points = inception_v3.inception_v3(Train_Images, num_classes)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = Train_Labels)

cross_entropy_mean = tf.reduce_mean(cross_entropy)

opt_op = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(cross_entropy_mean)

correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), Train_Labels)

acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accu')

# saver = tf.train.Saver()

init_op = tf.global_variables_initializer()

with tf.Session() as sess:

	sess.run(init_op)

	for i in range(200):

		img_names, labels = MyUtils.get_image_and_id_list(trainset_path)

		images, labels = MyUtils.get_random_samples(img_names, labels, batch_size, trainset_path, image_height, image_width)

		sess.run(opt_op, feed_dict={Train_Images: images, Train_Labels: labels})

		if i % 10 == 0:

			print "Setp: ", i, "Accuracy: ", sess.run(acc, feed_dict={Train_Images: images, Train_Labels: labels})
			print "Setp: ", i, "Loss: ", sess.run(cross_entropy_mean, feed_dict={Train_Images: images, Train_Labels: labels})
	
	# save_model = saver.save(sess, "save/model.ckpt") # need to correct