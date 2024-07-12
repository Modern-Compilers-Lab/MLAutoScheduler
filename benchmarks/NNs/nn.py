import tensorflow as tf

class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = tf.keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(None, None, 1))
        self.conv2 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(120, activation='relu')
        self.fc2 = tf.keras.layers.Dense(84, activation='relu')
        self.fc3 = tf.keras.layers.Dense(10)

    def call(self, x):
        x = tf.keras.layers.MaxPooling2D((2, 2))(self.conv1(x))
        x = tf.keras.layers.MaxPooling2D(2)(self.conv2(x))
        x = self.flatten(x)
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
dummy_input = np.random.rand(1, 28, 28, 1).astype(np.float32)


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
