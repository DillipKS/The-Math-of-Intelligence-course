# Optimize a polynomial function to find minima/maxima using 2nd order Newton's method.

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

# Function is: f(x) = 6x^5 - 5x^4 - 4x^3 + 3x^2

def f(x):
    return 6*x**5-5*x**4-4*x**3+3*x**2

def df(x):
    return 30*x**4-20*x**3-12*x**2+6*x

def d2f(x):
    return 120*x**3-60*x**2-24*x+6

def delta_fn(x):
    return abs(f(x)-0)

def newton_root_finding(x0, epsilon):
    # Finding root of a function using Newton's Method

    x_root = np.zeros(len(x0))
    for i, x in enumerate(x0):
        delta = delta_fn(x)
        while delta > epsilon:
            x = x - f(x) / float(df(x))
            delta = delta_fn(x)
        x_root[i] = x

    return list(x_root)

def newton_optimization(x1, epsilon):
    # Finding minima/maxima using Newton's Method

    x_critical = np.zeros(len(x1))
    minmax = {}

    for i, x in enumerate(x1):
        n = 0
        delta = delta_fn(x)

        while delta > epsilon:
            x = x - df(x) / float(d2f(x))
            delta = delta_fn(x)
            n += 1
            if n == 10000:
                break
        x_critical[i] = x
    print "\nCritical points: "
    pprint(list(x_critical))

    for x in x_critical:
        if d2f(x) > 0:
            minmax[x] = 'minima'
        elif d2f(x) < 0:
            minmax[x] = 'maxima'
        elif d2f(x) == 0:
            minmax[x] = 'saddle'

    return minmax

def plot_graph():
    x_points = np.arange(-0.8,1.1,0.01)
    fn_points = [f(x) for x in x_points]
    plt.plot(x_points, fn_points, 'ro')
    plt.show()


def main():
    epsilon = 1e-10
    x0 = [-1, 0, 0.5, 1]

    x_root = newton_root_finding(x0,epsilon)
    print "\nRoots are: "
    pprint(x_root)

    x1 = [-0.5, 0.1, 0.5, 1.5, 3]
    minmax = newton_optimization(x1, epsilon)
    print "\nMinima/Maxima points are: "
    pprint(minmax)

    plot_graph()

if __name__ == '__main__':
    main()


'''
Result-
Roots are: 
[-0.7953336454431276, 0.0, 0.6286669787778999, 1.0]

Critical points: 
[-0.5889879839687899,
 -8.969446052179515e-10,
 0.39415736752083946,
 0.8614972831146172,
 0.861497283114617]

Minima/Maxima points are: 
{-0.5889879839687899: 'maxima',
 -8.969446052179515e-10: 'minima',
 0.39415736752083946: 'maxima',
 0.861497283114617: 'minima',
 0.8614972831146172: 'minima'}
'''