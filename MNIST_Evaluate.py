import tensorflow as tf
import tensorflow.examples.tutorials.mnist
import pickle
import sys
import MNIST_Generate
import numpy as np

data_dir = r'/tmp/tensorflow/mnist/input_data'


def display(image_array, value):
    import matplotlib.pyplot as plt
    size = int(np.sqrt(image_array.size))
    fig = plt.figure(1, figsize=(14, 7))
    fig.canvas.set_window_title(str(np.argmax(value)))
    plt.subplot(121)
    plt.imshow(image_array.reshape((size, size)), cmap=plt.cm.bone)
    plt.axis("off")
    plt.subplot(122)
    plt.xticks(np.arange(10))
    colors = ['b'] * 10
    colors[np.argmax(value)] = 'g'
    plt.bar(np.arange(10) - .5, np.transpose(value), width=1, color=colors)
    plt.show()


def unpickle_weights(output_file_path):
    with open(output_file_path, "r") as f:
        return pickle.load(f)

if __name__ == "__main__":
    mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets(data_dir, one_hot=True)
    weights = np.load(sys.argv[1])
    w_convolve1 = weights['w_convolve1']
    w_connect2 = weights['w_connect2']
    b_convolve1 = weights['b_convolve1']
    b_connect2 = weights['b_connect2']
    weights = {'Convolve1': tf.constant(w_convolve1),
               'Connect2': tf.constant(w_connect2)}
    biases = {'Convolve1': tf.constant(b_convolve1),
              'Connect2': tf.constant(b_connect2)}
    x1 = tf.placeholder(tf.float32, [None, 784])
    y = MNIST_Generate.generate_output(x1, weights, biases)
    y_ = tf.placeholder(tf.float32, [None, 10])
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    batch_xs, batch_ys = mnist.test.images, mnist.test.labels
    print(sess.run(accuracy, feed_dict={x1: batch_xs, y_: batch_ys}))

    scaled_output = tf.sigmoid(y) / tf.reduce_sum(tf.sigmoid(y))
    for i in range(10):
        display(batch_xs[i], sess.run(scaled_output, feed_dict={x1: batch_xs[i:i + 1]}))
