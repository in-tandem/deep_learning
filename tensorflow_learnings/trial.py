import tensorflow as tf 
import sys

print(tf.__version__)

## creating constants using tensorflow
## returns tensorflow.python.framework.ops.Tensor
constant_1_tensor = tf.constant('this is a constant')
constant_2_tensor = tf.constant('this is a constant', dtype = tf.string)
constant_3_tensor = tf.constant('this is a constant', dtype = tf.string, shape = [2])##creates a 2 element array
constant_4_tensor = tf.constant('this is a constant', dtype = tf.string, shape = [2, 2])##creates a 2*2 matrix
constant_5_tensor = tf.constant([1,2,3,], name = 'another_name') 
constant_6_tensor = tf.constant(2, shape = [2,3])  ## creates a 2*3 matrix of each value of 2

with tf.Session() as session:

    print(constant_1_tensor.eval(), type(constant_1_tensor.eval()))
    print(session.run(constant_1_tensor), type(session.run(constant_1_tensor))) ## prints b'this is a constant'
    print(session.run(constant_2_tensor)) ## prints b'this is a constant'
    print(session.run(constant_3_tensor)) ## prints [b'this is a constant' b'this is a constant']
    print(session.run(constant_4_tensor)) ## prints  2*2 matrix
    print(session.run(constant_5_tensor), type(session.run(constant_5_tensor))) ##[1 2 3] <class 'numpy.ndarray'>
    print(session.run(constant_6_tensor))

