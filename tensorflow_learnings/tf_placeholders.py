import tensorflow as tf 
import numpy as np 
## creates placeholders, and values to be fed using feed_dict method.
## placeholders cannot be evaluated everrrr

p1 = tf.placeholder(dtype = tf.int32, name ='some_p', shape = [1,2])


##placeholder without any shapes

p2 = tf.placeholder(dtype=tf.int32, shape = None) ##anything can be fed to this particular placeholder

x = tf.placeholder(dtype = tf.float32, shape = None, name = 'x')

print(p1.__class__)
with tf.Session() as session:

    # print(p1.eval()) ## will fail error: You must feed a value for placeholder tensor

    session.run(p1, feed_dict={p1: np.random.randn(1,2)})

    # print(session.run(p1)) ## why cant you do this?? well instead of this..maybe just print the feed_dict value

    session.run(p2, feed_dict = {p2 : np.random.randn(2,4)})

    session.run(p2, feed_dict = {p2 : np.random.randn(1,4)})


    session.run(x, feed_dict = {x : np.random.randn(1,4)})


