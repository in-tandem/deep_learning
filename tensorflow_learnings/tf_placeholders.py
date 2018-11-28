import tensorflow as tf 
import numpy as np 
## creates placeholders, and values to be fed using feed_dict method.
## placeholders cannot be evaluated everrrr

p1 = tf.placeholder(dtype = tf.int32, name ='some_p', shape = [1,2])

print(p1.__class__)
with tf.Session() as session:

    # print(p1.eval()) ## will fail error: You must feed a value for placeholder tensor

    session.run(p1, feed_dict={p1: np.random.randn(1,2)})

    # print(session.run(p1)) ## why cant you do this?? well instead of this..maybe just print the feed_dict value