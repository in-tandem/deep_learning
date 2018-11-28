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
constant_7_tensor = tf.constant('aaa', shape = [2,3])  ## creates a 2*3 matrix of each value of 2


## another way to create constants are tf.fill
constant_8_tensor = tf.fill(dims = [2,3], value = 3) ##dims are the dimensions.creates a matrix
constant_9_tensor = tf.fill(dims = [2], value = 3) ##dims are the dimensions, has to be a list. creates a array

##difference between tensor constant and tensor fill
## - tensor constant can support arbitrary values eg [1,'a', True] but not fill

##in the above linees we only initialized the nodes. to execute the same, we would
## need to run it in session
with tf.Session() as session:

    print(constant_1_tensor.eval(), type(constant_1_tensor.eval()))
    print(session.run(constant_1_tensor), type(session.run(constant_1_tensor))) ## prints b'this is a constant'
    print(session.run(constant_2_tensor)) ## prints b'this is a constant'
    print(session.run(constant_3_tensor)) ## prints [b'this is a constant' b'this is a constant']
    print(session.run(constant_4_tensor)) ## prints  2*2 matrix
    print(session.run(constant_5_tensor), type(session.run(constant_5_tensor))) ##[1 2 3] <class 'numpy.ndarray'>
    print(session.run(constant_6_tensor))
    print(session.run(constant_7_tensor), type(session.run(constant_7_tensor)))
    print(session.run(constant_8_tensor))
    print(session.run(constant_9_tensor))