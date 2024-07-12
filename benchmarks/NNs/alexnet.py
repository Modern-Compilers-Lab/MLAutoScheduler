import tensorflow as tf

import tensorflow as tf
class Net(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.features = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=11, strides=4, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
            tf.keras.layers.Conv2D(64, kernel_size=5, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
            tf.keras.layers.Conv2D(192, kernel_size=3, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, kernel_size=3, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, kernel_size=3, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=3, strides=2)
        ])
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(1000),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1000),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(1000)
        ])

    def call(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.classifier(x)
        return x



import numpy as np
import time


#@tf.function(jit_compile=True)
def run(A, model):
    return model(A)

# Generate random input data with the specified shape
input_data = np.random.rand(1, 256, 256, 3).astype(np.float32)

# Instantiate the model
model = Net()


for i in range(20):
    # Start timing
    start_time = tf.timestamp()

    # Perform matrix multiplication A x B
    result = run(input_data, model)

    # End timing
    end_time = tf.timestamp()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print("Time elapsed for matrix multiplication: {:.4f} seconds".format(elapsed_time.numpy()))

