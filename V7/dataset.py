from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pdb
import os

class Dateset():
	def __init__(self, sess, data_dir, batch_size=50):
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.sess = sess

		# make tensor
		tfrecord_list = os.listdir(self.data_dir)
		tfrecord_list = [os.path.join(self.data_dir, name) for name in tfrecord_list if name.endswith('tfrecords')]
		file_queue = tf.train.string_input_producer(tfrecord_list)
		#pdb.set_trace()
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(file_queue)
		
		features = tf.parse_single_example(serialized_example,features={
			'label': tf.FixedLenFeature([], tf.int64),
			'img' : tf.FixedLenFeature([], tf.string),
			'img_height': tf.FixedLenFeature([], tf.int64),
			'img_width': tf.FixedLenFeature([], tf.int64)
			})

		img = tf.decode_raw(features['img'], tf.uint8)
		img_height = tf.cast(features['img_height'], tf.int32)
		img_width = tf.cast(features['img_width'], tf.int32)
		img = tf.reshape(img, tf.stack([img_height, img_width, 3]))
		#img = tf.reshape(img, [128, 64, 3])
		#pdb.set_trace()
		img = tf.image.resize_images(img, [299, 299])
		label = features['label']
		self.imgs, self.labels = tf.train.batch([img, label],
			batch_size = self.batch_size,
			capacity = 3000,
			num_threads = 4
			)
		self.coord=tf.train.Coordinator()
		self.threads= tf.train.start_queue_runners(coord=self.coord, sess=self.sess)
		#self.imgs = img
		#self.labels = label

	def get_batch(self):
		#pdb.set_trace()
		ans = self.sess.run([self.imgs, self.labels])
		imgs, labels = ans
		imgs = (imgs/255 - 0.5) * 2
		return imgs, labels

	def close(self):
		self.coord.request_stop()
		self.coord.join(self.threads)