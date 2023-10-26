import tensorflow as tf
import numpy as np
import time
@tf.function(jit_compile=True)
def conv(input_tensor,filter_tensor,strides,dilations):
    return tf.nn.conv2d(input_tensor, filter_tensor, strides=strides, padding='VALID', dilations=dilations)
#tf.keras.backend.clear_session()
#tf.config.optimizer.set_jit(False) 
# Create a random 4D input tensor (batch_size, height, width, channels)
# Define input and filter tensors
input_tensor =  tf.constant(np.random.rand(3, 2048, 2048, 3), dtype=tf.float32)
filter_tensor = tf.constant(np.random.rand(3, 3, 3, 1), dtype=tf.float32)

# Define strides and dilations
strides = [1, 1, 1, 1]
dilations = [1, 1, 1, 1]


# Perform convolution
#conv_layer  = conv(input_tensor,filter_tensor,strides,dilations)

# Define the convolutional layer


for i in range(6):
# Start timing
    start_time = tf.timestamp()

    # Apply the convolution operation

    output_tensor = conv(input_tensor,filter_tensor,strides,dilations)

    # End timing
    end_time = tf.timestamp()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print("Time elapsed for Conv2D operation: {:.4f} seconds".format(elapsed_time.numpy()))

