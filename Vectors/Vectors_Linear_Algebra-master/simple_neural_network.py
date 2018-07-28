import numpy as np

# sigmoid function
def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])                  # 4x3
    
# output dataset            
y = np.array([[0,0,1,1]]).T                 # 4x1

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1        # 3x1 ranging [-1,1)

for iter in xrange(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))            # 4x1

    # how much did we miss?
    l1_error = y - l1

    # element wise multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)   # 4x1

    # update weights with backpropagation
    syn0 += np.dot(l0.T,l1_delta)           # 3x1

print "Output After Training: \n", l1
print "Weight values: \n", syn0


'''
Output After Training: 
[[0.00966449]
 [0.00786506]
 [0.99358898]
 [0.99211957]]
Weight values: 
[[ 9.67299303]
 [-0.2078435 ]
 [-4.62963669]]
'''