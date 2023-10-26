import tensorflow as tf
import numpy as np
import time


@tf.function(jit_compile=True)
def matmul(A, B):
    return tf.matmul(A, B)
# Create random input tensors A and B
A = tf.constant(np.random.rand(1200, 1500), dtype=tf.float32)
B = tf.constant(np.random.rand(1500, 1000), dtype=tf.float32)





for i in range(6):
    # Start timing
    start_time = tf.timestamp()

    # Perform matrix multiplication A x B
    result = matmul(A, B)

    # End timing
    end_time = tf.timestamp()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print("Time elapsed for matrix multiplication: {:.4f} seconds".format(elapsed_time.numpy()))
