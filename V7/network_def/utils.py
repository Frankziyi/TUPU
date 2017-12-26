import tensorflow as tf


slim = tf.contrib.slim


initializers = {
    0: slim.variance_scaling_initializer(),
    1: slim.xavier_initializer(),
    2: slim.variance_scaling_initializer(),
    3: slim.variance_scaling_initializer()
}


def add_l2_constraint_layer(feature):
    feature = tf.nn.l2_normalize(feature, 1)
    alpha = tf.get_variable(
        "l2_softmax_alpha", shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(50))
    tf.add_to_collection("l2_softmax_alpha", alpha)
    feature *= alpha
    return feature


def normalized_fully_connected(inputs, num_outputs, scope,
                               activation_fn=None,
                               initializer=slim.variance_scaling_initializer()):
    assert len(inputs.shape) == 2, "inputs were not flatten"
    num_inputs = inputs.shape[1]
    weights = tf.get_variable("linear_weights",
                              shape=[num_inputs, num_outputs],
                              dtype=tf.float32,
                              initializer=initializer)
    weights = tf.transpose(weights, [1, 0])
    weights = tf.nn.l2_normalize(weights, 1, name="normalized_weights")
    weights = tf.transpose(weights, [1, 0])
    tf.add_to_collection("normalized_weights", weights)
    outputs = tf.matmul(inputs, weights)

    if activation_fn:
        outputs = activation_fn(outputs)

    return outputs


def add_se_block(inputs, r=16):
    pooling_window = inputs.shape[1:-1]
    inputs_depth = int(inputs.get_shape()[-1])
    net = slim.avg_pool2d(inputs, kernel_size=pooling_window,
                          padding="valid")
    net = slim.flatten(net)
    net = slim.fully_connected(net, int(inputs_depth / r),
                               activation_fn=tf.nn.relu,
                               scope="se_block_squeeze")
    net = slim.fully_connected(net, inputs_depth,
                               activation_fn=tf.sigmoid,
                               scope="se_block_extract")
    net = tf.reshape(net, [-1, 1, 1, inputs_depth])
    net = inputs * net
    return net


def get_asoftmax_loss(logits, endpoints, labels, num_labels, batch_size):
    # Gather the ground truth weights for each image and compute the acos in a
    # batch fashion
    # First transposing the weights to index a column of the original weights
    normalized_weights = tf.get_collection("normalized_weights")[0]
    transposed_weights = tf.transpose(normalized_weights, [1, 0])
    ground_truth_weights = tf.gather(transposed_weights, labels)

    # Compute phis
    features = endpoints["fc5"]
    l2_features = tf.nn.l2_normalize(features, 1)
    cos_batch = tf.reduce_sum(tf.multiply(l2_features, ground_truth_weights), 1,
                              keep_dims=True)
    thetas = tf.acos(cos_batch)
    m = tf.get_collection("asoftmax_m")[0]
    k = tf.floor(m * thetas / 3.1415926)
    phis = ((-1) ** k) * tf.cos(m * thetas) - 2 * k

    logits = endpoints["Logits"]

    gt_idx = tf.stack([range(batch_size), labels], axis=1)
    gt_logits = tf.squeeze(tf.gather_nd(logits, gt_idx))
    gt_logits = tf.reshape(gt_logits, (batch_size, 1))

    # compute ||x||
    feature_dist = tf.sqrt(tf.reduce_sum(features * features, axis=1))
    feature_dist = tf.reshape(feature_dist, (batch_size, 1))

    # Formulate A-Softmax loss
    # numerator (lamb * y + phis(x) * ||x||) / (1 + lamb)
    lamb = tf.get_collection("asoftmax_lambda")[0]
    numer = (lamb * gt_logits + phis * feature_dist) / (1.0 + lamb)

    # Normalize the logits for numerical stability
    norm = tf.reduce_max(logits, 1, keep_dims=True)
    logits -= norm
    numer -= norm
    gt_logits = tf.squeeze(tf.gather_nd(logits, gt_idx))
    gt_logits = tf.reshape(gt_logits, (batch_size, 1))

    # numer
    exp_numer = tf.exp(numer)

    # denom
    exp_logits = tf.exp(logits)
    exp_logits_sum = tf.reduce_sum(exp_logits, axis=1)
    exp_logits_sum = tf.reshape(exp_logits_sum, (batch_size, 1))
    exp_denom = exp_logits_sum - tf.exp(gt_logits) + exp_numer

    loss = tf.reduce_mean(
            -tf.log(exp_numer / exp_denom), name="a-softmax-mean")
    tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
