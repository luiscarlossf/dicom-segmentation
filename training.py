from tensorflow import nn, argmax, estimator, layers, keras, reshape, metrics, losses, train
import numpy as np

"""
def ccn_model_fn(images, labels, mode):

    #Input Images
    input_images = tf.reshape(images, [-1, 30, 30, 1])

    #Convolution layer with 20 filters of size 5x5
    conv1 = tf.layers.conv2d(
        inputs=input_images,python
        filters=20,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    #Pooling layer with max-pooling function of size 2x2
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)

    #Convolution layer with 50 filters of size 5x5
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=50,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    #Pooling layer with max-pooling function of size 2x2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=1)

    #Fully connected layer composed of an input layer with the number of pixels of
    #the images prior to that layer, followed by an activation layer with ReLU
    #functions, and an output layer with a softmax function giving the probability of
    #the input to belong to either class (spinal cord and non-spinal cord).

    pool2_flat = tf.reshape(pool2, [-1, 7 * 8 * 50])

    dense = tf.layers.dense(inputs=pool2_flat, units=2800, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=2)
    predictions = {
        "classes" : tf.argmax(input=logits, axis=1),
        "probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
"""

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=nn.relu)

    # Pooling Layer #1
    pool1 = layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=nn.relu)
    pool2 = layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = reshape(pool2, [-1, 7 * 7 * 64])
    dense = layers.dense(inputs=pool2_flat, units=1024, activation=nn.relu)
    dropout = layers.dropout(
        inputs=dense, rate=0.4, training=mode == estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": nn.softmax(logits, name="softmax_tensor")
    }

    if mode == estimator.ModeKeys.PREDICT:
        return estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == estimator.ModeKeys.TRAIN:
        optimizer = train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=train.get_global_step())
        return estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": metrics.accuracy(
        labels=labels, predictions=predictions["classes"])
    }
    return estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":
    # Load training and eval data
    ((train_data, train_labels),
     (eval_data, eval_labels)) = keras.datasets.mnist.load_data()

    train_data = train_data / np.float32(255)
    train_labels = train_labels.astype(np.int32)  # not required

    eval_data = eval_data / np.float32(255)
    eval_labels = eval_labels.astype(np.int32)  # not required

    # Create the Estimator
    mnist_classifier = estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}

    logging_hook = train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # train one step and display the probabilties
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1,
        hooks=[logging_hook])

    mnist_classifier.train(input_fn=train_input_fn, steps=1000)

    eval_input_fn = estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)




