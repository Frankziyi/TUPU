import tensorflow as tf
import tensorflow.contrib.slim as slim
from network_def import inception_reid
from network_def import inception_v3
from network_def import utils
import numpy as np
import os
import get_label
import get_random_samples
import Partion_TrainSet_And_ValidationSet
import get_images

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

dataset_path = '../../world/data-gpu-94/sysu-reid/person-reid-data/Market-1501/Market-1501-v15.09.15/'
trainset_path = dataset_path + '/bounding_box_train/' # 751 id
gallery_path = dataset_path + '/bounding_box_test/' # 750 id
probe_path = dataset_path + '/query/'

batch_size = 8
validation_batch_size = 32
num_classes = 1501
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999

data_node = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
labels_node = tf.placeholder(tf.int32, shape=None)

va_data_node = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
va_labels_node = tf.placeholder(tf.int32, shape=None)


# To know if it is training or not
train_flag = tf.placeholder(tf.bool)

# logits, end_points = inception_reid.inception_reid(Train_Images, num_classes, asoftmax=False)

logits, end_points = inception_v3.inception_v3(data_node, num_classes, is_training=train_flag)

variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionV3')

# variables_to_restore = [var for var in variables_to_restore if not var.name.startswith('InceptionV3/logits')]

saver_to_load = tf.train.Saver(variables_to_restore)

saver_to_restore = tf.train.Saver()

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels_node)

cross_entropy_mean = tf.reduce_mean(cross_entropy)

opt_op = tf.train.AdamOptimizer(learning_rate, beta1, beta2).minimize(cross_entropy_mean)

# opt_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_mean)

va_logits, va_end_points = inception_v3.inception_v3(va_data_node, num_classes, is_training=train_flag, reuse=True)

prediction = tf.cast(tf.argmax(va_logits, 1), tf.int32)

correct_prediction = tf.equal(prediction, va_labels_node)

acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accu')

init_op = tf.global_variables_initializer()

with tf.Session() as sess:

	sess.run(init_op)

	saver_to_load.restore(sess, "inception_v3.ckpt")

	TrainSet, ValidationSet = Partion_TrainSet_And_ValidationSet.Partion_TrainSet_And_ValidationSet(trainset_path, 0.9)

	for i in range(2000):

		TrainSet_Labels = get_label.get_label(TrainSet)

		images, labels = get_random_samples.get_random_samples(TrainSet, TrainSet_Labels, batch_size, trainset_path)

		sess.run(opt_op, feed_dict={data_node: images, labels_node: labels, train_flag: True})

		if i % 100 == 0:

			ValidationSet_Labels =get_label.get_label(ValidationSet)

			va_images, va_labels = get_random_samples.get_random_samples(ValidationSet, ValidationSet_Labels, validation_batch_size, trainset_path)

			print "Setp: ", i, "Accuracy: ", sess.run(acc, feed_dict={va_data_node: va_images, va_labels_node: va_labels, train_flag: False})
