import tensorflow as tf

import tensorflow as tf

class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        # Larger convolutional layer: 3 input image channel, 64 output channels, 7x7 square convolution kernel
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=5, padding='same', activation='relu')
        # Max pooling over a (2, 2) window
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        # Flatten the tensor before passing to fully connected layers
        self.flatten = tf.keras.layers.Flatten()
        # Fully connected layers
        self.fc1 = tf.keras.layers.Dense(120, activation='relu')
        self.fc2 = tf.keras.layers.Dense(84, activation='relu')
        self.fc3 = tf.keras.layers.Dense(10)

    def call(self, inputs):
        # Convolutional layers
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        # Flatten the tensor before passing to fully connected layers
        x = self.flatten(x)
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x



import numpy as np
import time


#@tf.function(jit_compile=True)
def run(A, model):
    return model(A)

# Create a dummy input tensor
dummy_input = np.random.rand(32, 230, 230, 3).astype(np.float32)


# Instantiate the model
model = Net()


for i in range(20):
    # Start timing
    start_time = tf.timestamp()

    # Perform matrix multiplication A x B
    result = run(dummy_input, model)

    # End timing
    end_time = tf.timestamp()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print("Time elapsed for matrix multiplication: {:.4f} seconds".format(elapsed_time.numpy()))
