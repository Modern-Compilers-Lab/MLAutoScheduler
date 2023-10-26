import tensorflow as tf
import numpy as np
import time
@tf.function(jit_compile=True)
def conv(input_tensor,conv_layer):
    return conv_layer(input_tensor)
#tf.keras.backend.clear_session()
#tf.config.optimizer.set_jit(False) 
# Create a random 4D input tensor (batch_size, height, width, channels)
input_tensor = tf.constant(np.random.rand(1, 2048, 3), dtype=tf.float32)

# Define the convolutional layer
conv_layer = tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding='same')

for i in range(6):
# Start timing
    start_time = tf.timestamp()

    # Apply the convolution operation

    output_tensor = conv(input_tensor,conv_layer)

    # End timing
    end_time = tf.timestamp()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print("Time elapsed for Conv2D operation: {:.4f} seconds".format(elapsed_time.numpy()))


