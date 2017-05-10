import tensorflow as tf
import tensorflow.examples.tutorials.mnist
import numpy as np
import pickle
import sys
data_dir = r'/tmp/tensorflow/mnist/input_data'


def max_pool(tensor):
    return tf.nn.max_pool(tensor, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def convolve(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def generate_output(x1, weights, biases):
    x1_reshaped = tf.reshape(x1, [-1, 28, 28, 1])

    x2 = convolve(x1_reshaped, weights['Convolve1'], biases['Convolve1'])
    x2_reshaped = tf.reshape(max_pool(x2), [-1, 14 * 14 * 32])

    y = tf.matmul(x2_reshaped, weights['Connect2']) + biases['Connect2']
    return y


def pickle_weights(output_file_path, weights):
    """
    Pickles weights into output_file_path
    Weights should be a list, formatted as
    [weight_first_layer, weight_second_layer..., bias_first_layer, bias_second_layer...]
    """
    with open(output_file_path, "w") as f:
        pickle.dump(weights, f)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(len(sys.argv))
        raise IOError("Incorrect number of arguments")
    save_file = sys.argv[1]
    if save_file[-4:] == ".npz":
        save_file += ".npz"
    mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets(data_dir, one_hot=True)

    weights = {'Convolve1': tf.Variable(tf.random_normal(shape=[5, 5, 1, 32], stddev=0.1)),
               'Connect2': tf.Variable(tf.random_normal(shape=[14 * 14 * 32, 10], stddev=0.1))}
    biases = {'Convolve1': tf.Variable(tf.random_normal(shape=[32], stddev=0.1)),
              'Connect2': tf.Variable(tf.random_normal(shape=[10], stddev=0.1))}

    x1 = tf.placeholder(tf.float32, [None, 784])
    y = generate_output(x1, weights, biases)
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    for _ in range(300):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x1: batch_xs, y_: batch_ys})

    print("Saving weights to %s" % sys.argv[1])
    np.savez_compressed(sys.argv[1],
                        w_convolve1=sess.run(weights["Convolve1"]),
                        w_connect2=sess.run(weights["Connect2"]),
                        b_convolve1=sess.run(biases["Convolve1"]),
                        b_connect2=sess.run(biases["Connect2"]))
