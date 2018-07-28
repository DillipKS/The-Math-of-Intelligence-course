# https://github.com/llSourcell/Second_Order_Optimization_Newtons_Method/blob/master/newtons_method_root_finding.py
"""
Newton's method
Author: Daniel Homola
Licence: BSD 3-clause
"""

from scipy.optimize import newton
from sklearn.utils.testing import assert_almost_equal

def f(x):
    return 6*x**5-5*x**4-4*x**3+3*x**2

def df(x):
    return 30*x**4-20*x**3-12*x**2+6*x

def dx(f, x):
    # Distance between x-axis and function value at x
    return abs(0-f(x))
    
#def newtons_method(f, df, x0, e, print_res=False):
def newtons_method(x0, e, print_res=False):
    delta = dx(f, x0)
    n = 0
    while delta > e:
        x0 = x0 - f(x0)/float(df(x0))
        delta = dx(f, x0)
        n += 1
        if n == 1000:
            break

    if print_res:
        print 'Error: ', delta
        print 'Root is at: ', x0
        print 'f(x) at root is: ', f(x0)
        print 'No. of iterations: ', n
    return x0

#def test_with_scipy(f, df, x0s, e):
def test_with_scipy(x0s, e):
    for x0 in x0s:
        #my_newton = newtons_method(f, df, x0, e)
        my_newton = newtons_method(x0, e)
        scipy_newton = newton(f, x0, df, tol=e)
        assert_almost_equal(my_newton, scipy_newton, decimal=5)
        print 'Tests passed.'

if __name__ == '__main__':
    # run test
    x0s = [0, .4, 200]    
    #test_with_scipy(f, df, x0s, 1e-5)
    test_with_scipy(x0s, 1e-5)
        
    for x0 in x0s:
        #newtons_method(f, df, x0, 1e-10, True)
        newtons_method(x0, 1e-10, True)
