from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
# => tf.Tensor([[1 2]
#               [3 4]], shape=(2, 2), dtype=int32)

# Broadcasting support
b = tf.add(a, 1)
print(b)
# => tf.Tensor([[2 3]
#               [4 5]], shape=(2, 2), dtype=int32)

# Operator overloading is supported
print(a * b)
# => tf.Tensor([[ 2  6]
#               [12 20]], shape=(2, 2), dtype=int32)

# Use NumPy values
import numpy as np

c = np.multiply(a, b)
print(c)
# => [[ 2  6]
#     [12 20]]

# Obtain numpy value from a tensor:
print(a.numpy())
# => [[1 2]
#     [3 4]]
with tf.device('/device:GPU:0'):
    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
    print(dataset1.output_shapes)  # ==> "(10,)"
    print(dataset1)  # ==> "tf.float32"

    dataset2 = tf.data.Dataset.from_tensor_slices(
    (tf.random_uniform([4]),
        tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))

    print(dataset2.output_shapes)  # ==> "((), (100,))"
    print(dataset2)  # ==> "(tf.float32, tf.int32)"

    dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
    print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"