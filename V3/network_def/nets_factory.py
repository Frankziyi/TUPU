# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from . import alexnet
from . import cifarnet
from . import inception
from . import lenet
from . import overfeat
from . import resnet_v1
from . import resnet_v2
from . import vgg
from . import densenet
from . import mobilenet
from . import mobilenet_lib
from . import mobilenet_v1
from . import xception
from . import resnet_face
from . import inception_reid
slim = tf.contrib.slim

networks_map = {"alexnet_v2": alexnet.alexnet_v2,
                "cifarnet": cifarnet.cifarnet,
                "overfeat": overfeat.overfeat,
                "vgg_a": vgg.vgg_a,
                "vgg_16": vgg.vgg_16,
                "vgg_19": vgg.vgg_19,
                "inception_v1": inception.inception_v1,
                "inception_v2": inception.inception_v2,
                "inception_v3": inception.inception_v3,
                "inception_v4": inception.inception_v4,
                "inception_resnet_v2": inception.inception_resnet_v2,
                "lenet": lenet.lenet,
                "resnet_v1_50": resnet_v1.resnet_v1_50,
                "resnet_v1_101": resnet_v1.resnet_v1_101,
                "resnet_v1_152": resnet_v1.resnet_v1_152,
                "resnet_v1_200": resnet_v1.resnet_v1_200,
                "resnet_v2_50": resnet_v2.resnet_v2_50,
                "resnet_v2_101": resnet_v2.resnet_v2_101,
                "resnet_v2_152": resnet_v2.resnet_v2_152,
                "resnet_v2_200": resnet_v2.resnet_v2_200,
                "densenet": densenet.densenet,
                "mobilenet": mobilenet.mobilenet,
                "mobilenet_50": mobilenet.mobilenet_50,
                "mobilenet_75": mobilenet.mobilenet_75,
                "mobilenet_wm_50_rm_224": mobilenet.mobilenet_wm_50_rm_224,
                "mobilenet_wm_75_rm_224": mobilenet.mobilenet_wm_75_rm_224,
                "mobilenet_wm_100_rm_224": mobilenet.mobilenet_wm_100_rm_224,
                "mobilenet_wm_50_rm_160": mobilenet.mobilenet_wm_50_rm_160,
                "mobilenet_wm_75_rm_160": mobilenet.mobilenet_wm_75_rm_160,
                "mobilenet_wm_100_rm_160": mobilenet.mobilenet_wm_100_rm_160,
                "mobilenet_wm_50_rm_128": mobilenet.mobilenet_wm_50_rm_128,
                "mobilenet_wm_75_rm_128": mobilenet.mobilenet_wm_75_rm_128,
                "mobilenet_wm_100_rm_128": mobilenet.mobilenet_wm_100_rm_128,
                "mobilenet_v1_100_224": mobilenet_lib.mobilenet_zoo["mobilenet_v1_100_224"],
                "mobilenet_v1_100_192": mobilenet_lib.mobilenet_zoo["mobilenet_v1_100_192"],
                "mobilenet_v1_100_160": mobilenet_lib.mobilenet_zoo["mobilenet_v1_100_160"],
                "mobilenet_v1_100_128": mobilenet_lib.mobilenet_zoo["mobilenet_v1_100_128"],
                "mobilenet_v1_75_224": mobilenet_lib.mobilenet_zoo["mobilenet_v1_75_224"],
                "mobilenet_v1_75_192": mobilenet_lib.mobilenet_zoo["mobilenet_v1_75_192"],
                "mobilenet_v1_75_160": mobilenet_lib.mobilenet_zoo["mobilenet_v1_75_160"],
                "mobilenet_v1_75_128": mobilenet_lib.mobilenet_zoo["mobilenet_v1_75_128"],
                "mobilenet_v1_50_224": mobilenet_lib.mobilenet_zoo["mobilenet_v1_50_224"],
                "mobilenet_v1_50_192": mobilenet_lib.mobilenet_zoo["mobilenet_v1_50_192"],
                "mobilenet_v1_50_160": mobilenet_lib.mobilenet_zoo["mobilenet_v1_50_160"],
                "mobilenet_v1_50_128": mobilenet_lib.mobilenet_zoo["mobilenet_v1_50_128"],
                "mobilenet_v1_25_224": mobilenet_lib.mobilenet_zoo["mobilenet_v1_25_224"],
                "mobilenet_v1_25_192": mobilenet_lib.mobilenet_zoo["mobilenet_v1_25_192"],
                "mobilenet_v1_25_160": mobilenet_lib.mobilenet_zoo["mobilenet_v1_25_160"],
                "mobilenet_v1_25_128": mobilenet_lib.mobilenet_zoo["mobilenet_v1_25_128"],
                "xception": xception.xception,
                "resnet_face": resnet_face.resnet_face,
                "resnet_face_wobn": resnet_face.resnet_face_wobn,
                "resnet_face_se": resnet_face.resnet_face_se,
                "inception_reid": inception_reid.inception_reid,
                "inception_reid_se": inception_reid.inception_reid_se,
               }

arg_scopes_map = {"alexnet_v2": alexnet.alexnet_v2_arg_scope,
                  "cifarnet": cifarnet.cifarnet_arg_scope,
                  "overfeat": overfeat.overfeat_arg_scope,
                  "vgg_a": vgg.vgg_arg_scope,
                  "vgg_16": vgg.vgg_arg_scope,
                  "vgg_19": vgg.vgg_arg_scope,
                  "inception_v1": inception.inception_v3_arg_scope,
                  "inception_v2": inception.inception_v3_arg_scope,
                  "inception_v3": inception.inception_v3_arg_scope,
                  "inception_v4": inception.inception_v4_arg_scope,
                  "inception_resnet_v2":
                  inception.inception_resnet_v2_arg_scope,
                  "lenet": lenet.lenet_arg_scope,
                  "resnet_v1_50": resnet_v1.resnet_arg_scope,
                  "resnet_v1_101": resnet_v1.resnet_arg_scope,
                  "resnet_v1_152": resnet_v1.resnet_arg_scope,
                  "resnet_v1_200": resnet_v1.resnet_arg_scope,
                  "resnet_v2_50": resnet_v2.resnet_arg_scope,
                  "resnet_v2_101": resnet_v2.resnet_arg_scope,
                  "resnet_v2_152": resnet_v2.resnet_arg_scope,
                  "resnet_v2_200": resnet_v2.resnet_arg_scope,
                  "densenet": densenet.densenet_arg_scope,
                  "mobilenet": mobilenet.mobilenet_arg_scope,
                  "mobilenet_50": mobilenet.mobilenet_arg_scope,
                  "mobilenet_75": mobilenet.mobilenet_arg_scope,
                  "mobilenet_wm_50_rm_224": mobilenet.mobilenet_arg_scope,
                  "mobilenet_wm_75_rm_224": mobilenet.mobilenet_arg_scope,
                  "mobilenet_wm_100_rm_224": mobilenet.mobilenet_arg_scope,
                  "mobilenet_wm_50_rm_160": mobilenet.mobilenet_arg_scope,
                  "mobilenet_wm_75_rm_160": mobilenet.mobilenet_arg_scope,
                  "mobilenet_wm_100_rm_160": mobilenet.mobilenet_arg_scope,
                  "mobilenet_wm_50_rm_128": mobilenet.mobilenet_arg_scope,
                  "mobilenet_wm_75_rm_128": mobilenet.mobilenet_arg_scope,
                  "mobilenet_wm_100_rm_128": mobilenet.mobilenet_arg_scope,
                  "mobilenet_v1_100_224": mobilenet_v1.mobilenet_v1_arg_scope,
                  "mobilenet_v1_100_192": mobilenet_v1.mobilenet_v1_arg_scope,
                  "mobilenet_v1_100_160": mobilenet_v1.mobilenet_v1_arg_scope,
                  "mobilenet_v1_100_128": mobilenet_v1.mobilenet_v1_arg_scope,
                  "mobilenet_v1_75_224": mobilenet_v1.mobilenet_v1_arg_scope,
                  "mobilenet_v1_75_192": mobilenet_v1.mobilenet_v1_arg_scope,
                  "mobilenet_v1_75_160": mobilenet_v1.mobilenet_v1_arg_scope,
                  "mobilenet_v1_75_128": mobilenet_v1.mobilenet_v1_arg_scope,
                  "mobilenet_v1_50_224": mobilenet_v1.mobilenet_v1_arg_scope,
                  "mobilenet_v1_50_192": mobilenet_v1.mobilenet_v1_arg_scope,
                  "mobilenet_v1_50_160": mobilenet_v1.mobilenet_v1_arg_scope,
                  "mobilenet_v1_50_128": mobilenet_v1.mobilenet_v1_arg_scope,
                  "mobilenet_v1_25_224": mobilenet_v1.mobilenet_v1_arg_scope,
                  "mobilenet_v1_25_192": mobilenet_v1.mobilenet_v1_arg_scope,
                  "mobilenet_v1_25_160": mobilenet_v1.mobilenet_v1_arg_scope,
                  "mobilenet_v1_25_128": mobilenet_v1.mobilenet_v1_arg_scope,
                  "xception": xception.xception_arg_scope,
                  "resnet_face": resnet_face.resnet_face_argscope,
                  "resnet_face_wobn": resnet_face.resnet_face_argscope,
                  "resnet_face_se": resnet_face.resnet_face_argscope,
                  "inception_reid": inception_reid.inception_reid_arg_scope,
                  "inception_reid_se": inception_reid.inception_reid_arg_scope,
                 }


def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False,
                   **kwargs):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

  Args:
    name: The name of the network.
    num_classes: The number of classes to use for classification.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
        logits, end_points = network_fn(images)
  Raises:
    ValueError: If network `name` is not recognized.
  """
  if name not in networks_map:
    raise ValueError("Name of network unknown %s" % name)
  arg_scope = arg_scopes_map[name](weight_decay=weight_decay, **kwargs)
  func = networks_map[name]
  @functools.wraps(func)
  def network_fn(images, **kwargs_fn):
    with slim.arg_scope(arg_scope):
      return func(images, num_classes, is_training=is_training, **kwargs_fn)
  if hasattr(func, "default_image_size"):
    network_fn.default_image_size = func.default_image_size

  return network_fn
