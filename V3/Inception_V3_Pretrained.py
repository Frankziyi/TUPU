import tensorflow as tf
import tensorflow.contrib.slim as slim
from network_def import inception_reid
from network_def import inception_v3
from network_def import utils
import numpy as np
import os
import get_image_and_id_list
import get_random_samples
# import Partion_TrainSet_And_ValidationSet

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

dataset_path = '../../world/data-gpu-94/sysu-reid/person-reid-data/Market-1501/Market-1501-v15.09.15/'
trainset_path = dataset_path + '/bounding_box_train/' # 751 id
gallery_path = dataset_path + '/bounding_box_test/' # 750 id
probe_path = dataset_path + '/query/'

batch_size = 8
num_classes = 1501
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999

# Train_Images = tf.placeholder(tf.float32, shape=(batch_size, 128, 64, 3))
Train_Images = tf.placeholder(tf.float32, shape=(batch_size, 299, 299, 3))
Train_Labels = tf.placeholder(tf.int32, shape=batch_size)

# logits, end_points = inception_reid.inception_reid(Train_Images, num_classes, asoftmax=False)

logits, end_points = inception_v3.inception_v3(Train_Images, num_classes)

variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionV3')

# variables_to_restore = [var for var in variables_to_restore if not var.name.startswith('InceptionV3/logits')]

saver_to_load = tf.train.Saver(variables_to_restore)

saver_to_restore = tf.train.Saver()

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = Train_Labels)

cross_entropy_mean = tf.reduce_mean(cross_entropy)

opt_op = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(cross_entropy_mean)

# opt_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean)

correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), Train_Labels)

acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accu')

init_op = tf.global_variables_initializer()

with tf.Session() as sess:

	sess.run(init_op)

	saver_to_load.restore(sess, "inception_v3.ckpt")

	for i in range(2500):

		img_names, labels = get_image_and_id_list.get_image_and_id_list(trainset_path)

		images, labels = get_random_samples.get_random_samples(img_names, labels, batch_size, trainset_path)

		sess.run(opt_op, feed_dict={Train_Images: images, Train_Labels: labels})

		if i % 100 == 0:

			print "Setp: ", i, "Accuracy: ", sess.run(acc, feed_dict={Train_Images: images, Train_Labels: labels})

	save_model = saver_to_restore.save(sess, "save/Trained_Without_Validation.ckpt")
