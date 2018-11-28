import tensorflow as tf 
import numpy as np 


## by default a graph is created. eg in our previous modules, tf_variables or tf_constants.. even though
## we did not explicitly declare so a graph was created. the session ran also used the default graph

## we will create one explicit graph and use teh default one
## we will see if variables can be shared across each in a specific session
## run default session and try to access explicit graph tensors
## run explicit session and try to access default graph tensors

v1 = tf.Variable(name = 'outside_v1', dtype = tf.int32,  initial_value = [22])

g1 = tf.Graph()

with g1.as_default():
    v2 = tf.Variable(name = 'outside_v1', dtype = tf.int32,  initial_value = [22])
    init_op_g1 = tf.global_variables_initializer()


# Add an Op to initialize global variables.
init_op = tf.global_variables_initializer()

with tf.Session(graph = g1) as explicit_session:
    
    # explicit_session.run(init_op) ## this fails , as init_op is not within the graph. so the operation canot be executed

    explicit_session.run(init_op_g1)

    # explicit_session.run(v1) ## as expected fails

    explicit_session.run(v2)

with tf.Session() as implicit_session:

    implicit_session.run(init_op)

    # implicit_session.run(v2)## as expected fails

    implicit_session.run(v1)
