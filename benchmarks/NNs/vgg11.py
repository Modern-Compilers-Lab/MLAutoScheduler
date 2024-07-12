import tensorflow as tf

import tensorflow as tf

class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.features = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            
            tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            
            tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            
            tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            
            tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            
            tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            
            tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            
            tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        ])
    
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(1000, activation='relu'),
            
            tf.keras.layers.Dense(1000, activation='relu'),
            
            tf.keras.layers.Dense(1000),
        ])

    def call(self, inputs):
        x = self.features(inputs)
        x = self.avgpool(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.classifier(x)
        return x


import numpy as np
import time


@tf.function(jit_compile=True)
def run(A, model):
    return model(A)

# Generate random input data with the specified shape
input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)

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

