import tensorflow as tf
import numpy as np
import time
#@tf.function(jit_compile=True)
# Define a function for max pooling
def max_pool(input_tensor, ksize, strides):
    return tf.nn.max_pool(input_tensor, ksize=ksize, strides=strides, padding='VALID', data_format='NHWC')

input_tensor =  tf.constant(np.random.rand(1,  2048, 2048,32), dtype=tf.float32)  # NCHW layout
print (input_tensor.shape)
ksize = [3,3]  # Size of the pooling window
strides = [1, 1, 1, 1]  



for i in range(6):
# Start timing
    start_time = tf.timestamp()

    # Apply the convolution operation

    output_tensor = max_pool(input_tensor,ksize,strides)
    print(output_tensor.shape)
    # End timing
    end_time = tf.timestamp()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print("Time elapsed for Conv2D operation: {:.4f} seconds".format(elapsed_time.numpy()))

