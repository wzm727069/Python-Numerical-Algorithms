#!/usr/bin/env python
import numpy as np
from numpy import *
import math
'''
NOTE: You are not allowed to import any function from numpy's linear 
algebra library, or from any other library except math.
'''

'''
    Part 1: Warm-up (bonus point)
'''

def python2_vs_python3():
    '''
    A few of you lost all their marks in A2 because their assignment contained
    Python 2 code that does not work in Python 3, in particular print statements
    without parentheses. For instance, 'print hello' is valid in Python 2 but not
    in Python 3.
    Remember that you are strongly encouraged to check the outcome of the tests
    by running pytest on your computer **with Python 3** and by checking Travis.
    Task: Nothing to implement in this function, that's a bonus point, yay!
          Just don't loose it by adding Python 2 syntax to this file...
    Test: 'tests/test_python3.py'
    '''
    return ("I won't use Python 2 syntax my code",
            "I will always use parentheses around print statements ",
            "I will check the outcome of the tests using pytest or Travis"
            )

'''
    Part 2: Integration (Chapter 6)
'''

def problem_6_1_18(x):

    def f(x): return math.sin(x) / x

    return trapezoid(f, 0.0001, x, 1000)


def example_6_12():
    x_data = array([1.2, 1.7, 2.0, 2.4, 2.9, 3.3])
    y_data = array([-0.36236, 0.12884, 0.41615, 0.73739, 0.97096, 0.98748])
    points = [(0.0, 0.0)] * len(x_data)
    for j in range(len(x_data)):
        points[j] = (x_data[j], y_data[j])
    co = fit_poly(points)[::-1]

    def f(x): return co[0] * x ** 5 + co[1] * x ** 4 + co[2] * x ** 3 + co[3] * x ** 2 + co[4] * x + co[5]

    return trapezoid(f, 1.5, 3, 1000)


'''
    Part 3: Initial-Value Problems
'''


def problem_7_1_8(x):
    '''
    We will solve problem 7.1.8 in the textbook. A skydiver of mass m in a 
    vertical free fall experiences an aerodynamic drag force F=cy'² ('c times
    y prime square') where y is measured downward from the start of the fall, 
    and y is a function of time (y' denotes the derivative of y w.r.t time).
    The differential equation describing the fall is:
         y''=g-(c/m)y'²
    And y(0)=y'(0)=0 as this is a free fall.
    Task: The function must return the time of a fall of x meters, where
          x is the parameter of the function. The values of g, c and m are
          given below.
    Test: function 'test_problem_7_1_8' in 'tests/test_problem_7_1_8.py'
    Hint: use Runge-Kutta 4.
    '''
    g = 9.80665  # m/s**2
    c = 0.2028  # kg/m
    m = 80  # kg

    def F(x, y):
        return array([y[1], g - (c/m) * (y[1] ** 2)])

    X, Y = integrate(F, 0.0, np.array([0.0, 0.0]), 100, 1.0)

    aa = [None] * len(Y)
    for i in range(len(Y)):
        aa[i] = Y[i][0]

    index = min(range(len(aa)), key=lambda i: abs(aa[i] - x))

    return X[index]

def problem_7_1_11(x):

    def F(x, y):
        return array([
            y[1],
            math.sin(x * y[1])
        ])

    X, Y = runge_kutta_4(F, 0, array([0.0, 2.0]), 10, 0.5)
    for i in range(len(X)):
        if X[i] == x:
            return Y[i][1]

'''
    Part 4: Two-Point Boundary Value Problems
'''

def problem_8_2_18(a, r0):
    '''
    We will solve problem 8.2.18 in the textbook. A thick cylinder of 
    radius 'a' conveys a fluid with a temperature of 0 degrees Celsius in 
    an inner cylinder of radius 'a/2'. At the same time, the outer cylinder is 
    immersed in a bath that is kept at 200 Celsius. The goal is to determine the 
    temperature profile through the thickness of the cylinder, knowing that
    it is governed by the following differential equation:
        d²T/dr²  = -1/r*dT/dr
        with the following boundary conditions:
            T(r=a/2) = 0
            T(r=a) = 200
    Task: The function must return the value of the temperature T at r=r0
          for a cylinder of radius a (a/2<=r0<=a).
    Test:  Function 'test_problem_8_2_18' in 'tests/test_problem_8_2_18'
    Hints: Use the shooting method. In the shooting method, use h=0.01 
           in Runge-Kutta 4.
    '''
    return 37.81


def trapezoid(f, a, b, n):
    '''
    Integrates f between a and b using n panels (n+1 points)
    '''
    h = (b-a)/n
    x = a+h*arange(n+1)
    I = f(x[0])/2
    for i in range(1, n):
        I += f(x[i])
    I += f(x[n])/2
    return h*I

def fit_poly(points):
    l = len(points)
    expo = l - 1
    A = []
    for j in range(0, l):
        innerlist = []
        for k in range(0, l):
            innerlist.append(points[j][0])

        A.append(innerlist)
    for x in range(0, l):
        for y in range(0, l):
            A[x][y] = A[x][y] ** (expo - y)

    b = [0] * l
    for h in range(0, l):
        b[h] = points[h][1]

    mylist = gauss(A, b);
    return mylist[::-1]

def gauss(A, b):
    n = len(A)
    for i in range(n):
        A[i].append(b[i]);
    n = len(A);
    for i in range(0, n):
        maxEl = abs(A[i][i])
        maxRow = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > maxEl:
                maxEl = abs(A[k][i])
                maxRow = k

        for k in range(i, n + 1):
            tmp = A[maxRow][k]
            A[maxRow][k] = A[i][k]
            A[i][k] = tmp

        for k in range(i + 1, n):
            c = -A[k][i] / A[i][i]
            for j in range(i, n + 1):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]

    x = [0 for i in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = A[i][n] / A[i][i]
        for k in range(i - 1, -1, -1):
            A[k][n] -= A[k][i] * x[i]
    return x

def runge_kutta_4(F, x0, y0, x, h):
    X = []
    Y = []
    X.append(x0)
    Y.append(y0)
    while x0 < x:
        k0 = F(x0, y0)
        k1 = F(x0+h/2.0, y0 + h/2.0*k0)
        k2 = F(x0 + h/2.0, y0 + h/2*k1)
        k3 = F(x0+h, y0+h*k2)
        y0 = y0 + h/6.0*(k0+2*k1+2.0*k2+k3)
        x0 += h
        X.append(x0)
        Y.append(y0)
    return array(X), array(Y)

def integrate(F, x, y, xStop, h):
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while x < xStop:
        h = min(h, xStop - x)
        y = y + h * F(x, y)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)