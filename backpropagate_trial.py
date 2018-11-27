import numpy as np 


## i = 10 * 2
## h = 3
## w1 = 2 * 3
## z1 = 10 * 3
## w2 = 3 * 2

I = np.random.randint(low = 1, high = 9, size = (10,2))
w1 = np.random.rand(2,3)
z1 = np.random.rand(10,3)
theta_z1 = np.random.rand(10,3)
w2 = np.random.rand(3,2)
z2 = np.random.rand(10,2)
theta_z2 = np.random.rand(10,2)
e2 = np.random.rand(10,2)
eta = 0.012

layers = 3
w = [w1,w2]
z = [z1,z2]
theta = [I,theta_z1, theta_z2]

e = e2
for i in range(layers - 1)[::-1]: ##navigate in opp direction 
    
    old_weight = w[i]
    w[i ] = w[i] - eta *(np.dot(theta[i].T, e))
    e = np.dot(e, old_weight.T) * theta[i]


