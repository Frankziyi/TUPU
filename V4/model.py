from network_def import inception_v3
import sys
# Absolute path
# sys.path.insert(0, '/home/lixiang2/.local/lib/python2.7/site-packages')
# sys.path.insert(0, '/home/share/jiening/resnet/data')
import tensorflow as tf
import numpy as np
# import argparse

# parser = argparse.ArgumentParser(description='Define parameters.')

# # dataset
# from datasets import Dataset, root, join, RandomCrop, np, Batch_norm, Preprocessor, Resize, MultipleDatasets

# prep = Preprocessor(
#     sample_preprocess=RandomCrop(224, 224, resize_mode=(230, 230), meanvalue=np.array([102, 102, 101])[None, None, :], transpose = None),
# )



class pre_trained_inception_v3:
    def __init__(self, model_dir):
     
        # id_count = 5841    
        # id_count = 1025        
        self.batch = 50

        # self.model = ResNet([224, 224], True, id_count, batch_size = self.batch)
        self.inputs = tf.placeholder(tf.float32, shape=(self.batch, 299, 299, 3))

        _,  self.end_points = inception_v3.inception_v3(self.inputs, is_training=False, num_classes=1501, batch_size = self.batch)
        self.model_dir = model_dir
        # A TensorFlow Session for use in interactive contexts, such as a shell.
        self.sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        # saver.restore(self.sess, '/home/share/lixiang2/resnet/saves/')
        # saver.restore(self.sess, '/home/share/jiening/resnet/saves_lixiang/')
        saver.restore(self.sess, self.model_dir)
        print 'model load'

    def get_feature(self, images):
        
        # images = prep(images)
        return self.batch_eval(images)
    
    def batch_eval(self, x):
        ol = len(x)
        n = (len(x) / self.batch) + 1
        # nl = n * self.batch
        x = np.resize(x, [n, self.batch, 299, 299, 3])
        ans = []
        for i in x:
            feed_dict = {
                    self.inputs: i
                    }
            feature = self.sess.run([self.end_points['Feature']], feed_dict = feed_dict)
            # print(np.shape(feature))
            ans.append(feature)
            #ans.append(self.sess.run([self.end_points['Feature']], feed_dict = feed_dict)[0])

            # print(self.sess.run([self.model.is_training], feed_dict = feed_dict))
        print(np.shape(ans))
        ans = np.resize(ans, [ol, np.shape(ans)[5]])
        print(np.shape(ans))
        return ans


