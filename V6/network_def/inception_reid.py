import tensorflow as tf
import tensorflow.contrib.slim as slim

from activation_fns import prelu
import utils

# Inception + asoftmax
def inception_reid(inputs,
                   num_classes=1000,
                   is_training=True,
                   normalizer_fn=slim.batch_norm,
                   **kwargs):
    return inception_reid_base(inputs, num_classes=num_classes,
                               is_training=is_training,
                               normalizer_fn=normalizer_fn,
                               **kwargs)

# Inception + asoftmax + se
def inception_reid_se(inputs,
                      num_classes=1000,
                      is_training=True,
                      normalizer_fn=slim.batch_norm,
                      **kwargs):
    return inception_reid_base(inputs, num_classes=num_classes,
                               is_training=is_training,
                               normalizer_fn=normalizer_fn,
                               use_se_block=True,
                               **kwargs)


def inception_reid_base(inputs,
                        num_classes=1000,
                        is_training=True,
                        normalizer_fn=slim.batch_norm,
                        scope="inception_reid",
                        use_se_block=False,
                        **kwargs):
    # Resize image before entering the network
    inputs = tf.image.resize_images(inputs, (144, 56))

    with tf.variable_scope(scope) as sc:
        end_points_collection = sc.name + "_end_points"
        with slim.arg_scope([slim.separable_conv2d, slim.conv2d,
                             slim.avg_pool2d],
                            outputs_collections=[end_points_collection]), slim.arg_scope([normalizer_fn],
                                                                                         is_training=is_training):

            # entry flow
            net = slim.conv2d(inputs, 32, [3, 3], stride=1, padding="same",
                              activation_fn=prelu,
                              normalizer_fn=normalizer_fn,
                              scope="conv1")
            net = slim.conv2d(net, 32, [3, 3], stride=1, padding="same",
                              activation_fn=prelu,
                              normalizer_fn=normalizer_fn,
                              scope="conv2")
            net = slim.conv2d(net, 32, [3, 3], stride=1, padding="same",
                              activation_fn=prelu,
                              normalizer_fn=normalizer_fn,
                              scope="conv3")
            net = slim.max_pool2d(net, [2, 2], stride=2, padding="same",
                                  scope="pool1")

            # inception blocks
            net = _inception_block(net, 64, use_se_block=use_se_block,
                                   scope="inception_1a")
            net = _inception_block_downsampling(net, 64, scope="inception_1b")

            net = _inception_block(net, 128, use_se_block=use_se_block,
                                   scope="inception_2a")
            net = _inception_block_downsampling(net, 128,
                                                use_se_block=use_se_block,
                                                scope="inception_2b")

            net = _inception_block(net, 256, use_se_block=use_se_block,
                                   scope="inception_3a")
            net = _inception_block_downsampling(net, 256,
                                                use_se_block=use_se_block,
                                                scope="inception_3b")

            # logits
            end_points = {}
            with tf.variable_scope("Logits"):
                net = slim.avg_pool2d(net, [9, 4])
                net = slim.flatten(net)
                fc5 = slim.fully_connected(net, 256, activation_fn=None,
                                           scope="fc5")

                # return without constructing the huge classification layer
                # if we're using the network as feature extractor in testing
                if not is_training:
                    return fc5

                if kwargs["asoftmax"]:
                    # A-softmax requires the weights of the last fc layer to be
                    # l2-normalized
                    logits = utils.normalized_fully_connected(fc5, num_classes,
                                                              scope="Logits")
                else:
                    logits = slim.fully_connected(fc5, num_classes,
                                                  activation_fn=None,
                                                  scope="Logits")
                predictions = slim.softmax(logits, scope="Predictions")

                end_points["Logits"] = logits
                end_points["Predictions"] = predictions
                end_points["fc5"] = fc5

    return logits, end_points


def inception_reid_arg_scope(weight_decay=0.00001,
                             batch_norm_decay=0.9,
                             batch_norm_epsilon=0.001,
                             **kwargs):
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.batch_norm],
                            decay=batch_norm_decay,
                            epsilon=batch_norm_epsilon) as scope:
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                **kwargs) as scope:
                return scope


def _inception_block(inputs, output_channels, normalizer_fn=slim.batch_norm,
                     use_se_block=False, scope=None):
    with tf.variable_scope(scope):
        # Branch 1
        branch_1 = slim.conv2d(inputs, output_channels, [1, 1], stride=1,
                               padding="same",
                               activation_fn=prelu,
                               normalizer_fn=normalizer_fn)

        # Branch 2
        branch_2 = slim.conv2d(inputs, output_channels, [1, 1], stride=1,
                               padding="same",
                               activation_fn=prelu,
                               normalizer_fn=normalizer_fn)
        branch_2 = slim.conv2d(branch_2, output_channels, [3, 3], stride=1,
                               padding="same",
                               activation_fn=prelu,
                               normalizer_fn=normalizer_fn)

        # Branch 3
        branch_3 = slim.conv2d(inputs, output_channels, [1, 1], stride=1,
                               padding="same",
                               activation_fn=prelu,
                               normalizer_fn=normalizer_fn)
        branch_3 = slim.conv2d(branch_3, output_channels, [3, 3], stride=1,
                               padding="same",
                               activation_fn=prelu,
                               normalizer_fn=normalizer_fn)
        branch_3 = slim.conv2d(branch_3, output_channels, [3, 3], stride=1,
                               padding="same",
                               activation_fn=prelu,
                               normalizer_fn=normalizer_fn)

        # Branch 4
        branch_4 = slim.avg_pool2d(inputs, kernel_size=3, stride=1,
                                   padding="same")
        branch_4 = slim.conv2d(branch_4, output_channels, [1, 1], stride=1,
                               padding="same",
                               activation_fn=prelu,
                               normalizer_fn=normalizer_fn)

        # Depth concatenating all branches
        output = tf.concat([branch_1, branch_2, branch_3, branch_4], axis=3)

        if use_se_block:
            output = utils.add_se_block(output)

        return output


def _inception_block_downsampling(inputs, output_channels,
                                  normalizer_fn=slim.batch_norm,
                                  use_se_block=False,
                                  scope=None):
    with tf.variable_scope(scope):
        # Branch 1
        branch_1 = slim.conv2d(inputs, output_channels, [1, 1], stride=1,
                               padding="same",
                               activation_fn=prelu,
                               normalizer_fn=normalizer_fn)
        branch_1 = slim.conv2d(branch_1, output_channels, [3, 3], stride=2,
                               padding="same",
                               activation_fn=prelu,
                               normalizer_fn=normalizer_fn)

        # Branch 2
        branch_2 = slim.conv2d(inputs, output_channels, [1, 1], stride=1,
                               padding="same",
                               activation_fn=prelu,
                               normalizer_fn=normalizer_fn)
        branch_2 = slim.conv2d(branch_2, output_channels, [3, 3], stride=1,
                               padding="same",
                               activation_fn=prelu,
                               normalizer_fn=normalizer_fn)
        branch_2 = slim.conv2d(branch_2, output_channels, [3, 3], stride=2,
                               padding="same",
                               activation_fn=prelu,
                               normalizer_fn=normalizer_fn)

        # Branch 3
        branch_3 = slim.avg_pool2d(inputs, kernel_size=3, stride=2,
                                   padding="same")

        # Depth concatenating all branches
        output = tf.concat([branch_1, branch_2, branch_3], axis=3)

        if use_se_block:
            output = utils.add_se_block(output)

        return output
