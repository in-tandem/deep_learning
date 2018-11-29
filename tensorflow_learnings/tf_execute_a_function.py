import tensorflow as tf 
import numpy as np 

## y  = ax + b

graph = tf.Graph()

with graph.as_default() :
    
    x = tf.placeholder(dtype = tf.float32, shape = None, name = 'x')
    a = tf.placeholder(dtype = tf.float32, shape = None)

    b = tf.constant([4.], dtype= tf.float32)

    init_op = tf.global_variables_initializer()

# session = tf.Session(graph = graph)

with tf.Session(graph = graph) as session:
    
    session.run(init_op)

    print(session.run(tf.matmul(a,x) + b, feed_dict = {x : np.random.randn(2,5), a:np.random.randn(3,2)}))

    y = tf.add(tf.matmul(a,x) ,b )

    value = session.run(y, feed_dict = {x : np.random.randn(2,5), a:np.random.randn(3,2)})

    print(value, value.__class__)
