import tensorflow as tf 

## we will see how to declare variables and use them. 
## usage of get_variable
## we will also see how declaring and initialization of variables are separate operations

## type of v1 - tensorflow.python.ops.variables.RefVariable
v1 = tf.get_variable(name = 'v1', dtype=tf.int32, shape =1) 
v2 = tf.get_variable(name = 'v2', dtype=tf.int32, shape =[1]) ## you can see shape  = 1/[1] creates a 1d array
print(v1, v2, type(v1))
##create variable from another tensor
v3 = tf.get_variable(name ='from_another', dtype =tf.int32, initializer = tf.constant(2)) ##dtype has to be given ad should match with initializer


##creating variables using tf.Variable

v4 = tf.Variable(4)
v5 = tf.Variable(tf.constant('somak'))
v6 = tf.Variable(name = 'some_unique_name', initial_value=[1,2,3])

v8 = tf.Variable(name = 'some_unique_name_1', initial_value=v4)
##up until now we have only declared the variables. initialization(even when values are given) has not happened yet

# Add an Op to initialize global variables.
init_op = tf.global_variables_initializer()

with tf.Session() as session:

    
    session.run(init_op)
    print(session.run(v1)) ## if we do this without initilization, leads to FailedPreconditionError :  Attempting to use uninitialized value Variable
    ##we will see v1 is initialized to default value of int which is 0

    print(session.run(v4),session.run(v5),session.run(v6))

    v7 = v4

    ##the below leads to failure as initializations has already happened
    # v8 = tf.Variable(name = 'some_unique_name_1', initial_value=v4)

    print(session.run(v7), session.run(v8))

    assert(v7 == v4) ## success
    #assert(v8 == v4) ## failure


    ## how to use variables
    print(7 + session.run(v4), session.run(v4 + 7))

    print(v6.eval())

    v6.assign([11,22, 33]) ##must adhere to shape declared else it fails

    print(v6, session.run(v6)) ##ding ding ding!!! doest print 11,22,33 but instead 1,2,3

    v6 = v6.assign([11,22, 33])

    print(session.run(v6))