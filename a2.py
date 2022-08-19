#!/usr/bin/env python
from numpy import *
import numpy as np
import math

'''
NOTE: You are not allowed to import any function from numpy's linear 
algebra library, or from any other library except math.
'''

'''
    Part 1: Warm-up (bonus point)
'''

def spaces_and_tabs():
    '''
    A few of you lost all their marks in A1 because their file
    contained syntax errors such as:
      + Missing ':' at the end of 'if' or 'for' statements.
      + A mix of tabs and spaces to indent code. Remember that indendation is part
        of Python's syntax. Mixing tabs and spaces is not good because it
        makes indentation levels ambiguous. For this reason, files
        containing a mix of tabs and spaces are invalid in Python 3. This
        file uses *spaces* for indentation: don't add tabs to it!
    Remember that you are strongly encouraged to check the outcome of the tests
    by running pytest on your computer and by checking Travis.
    Task: Nothing to implement in this function, that's a bonus point, yay!
          Just don't loose it by adding syntax errors to this file...
    Test: 'tests/test_spaces_and_tabs.py'
    '''
    return ("I won't use tabs in my code",
            "I will make sure that my code has no syntax error",
            "I will check the outcome of the tests using pytest or Travis"
            )

'''
    Part 2: Linear regression
'''

def problem_3_2_5():
    year = arange(1994, 2010)  # from 1994 to 2009
    ppm = array([356.8, 358.2, 360.3, 361.8, 364.0, 365.7, 366.7, 368.2,
                 370.5, 372.2, 374.9, 376.7, 378.7, 381.0, 382.9, 384.7])
    xbar = mean(year)
    b = sum(ppm * (year - xbar)) / sum(year * (year - xbar))
    return b


def extrapolation_3_2_5():
    year = arange(1994, 2010)  # from 1994 to 2009
    ppm = array([356.8, 358.2, 360.3, 361.8, 364.0, 365.7, 366.7, 368.2,
                 370.5, 372.2, 374.9, 376.7, 378.7, 381.0, 382.9, 384.7])
    xbar = mean(year)
    ybar = mean(ppm)
    b = sum(ppm*(year-xbar))/sum(year*(year-xbar))
    a = ybar - xbar * b
    return (2020*b+a)


'''
    Part 3: Non-linear equations
'''

'''
    We will solve problem 4.1.19 in the textbook:
    "
        The speed v of a Saturn V rocket in vertical flight near the surface
        of earth can be approximated by:
            v = u*log(M0/(M0-mdot*t))-g*t
            (log base is e)
        where:
           * u = 2510 m /s is the velocity of exhaust relative to the rocket
           * M0 = 2.8E6 kg is the mass of the rocket at liftoff
           * mdot = 13.3E3 kg/s is the rate of fuel consumption
           * g = 9.81 m/s**2 is the gravitational acceleration
           * t is the time measured from liftoff
    "
'''
def f_and_df(t):
    u=2510
    M0=2.8E6
    mdot=13.3E3
    g=9.81
    def f(t): return u * math.log(M0 / (M0 - mdot * t)) - g * t

    def f_1(f, t, h=10E-4):
        return (f(t + h) - f(t - h)) / (2 * h)

    return f(t), f_1(f, t)

def problem_4_1_19(v1, acc):
    u = 2510
    M0 = 2.8E6
    mdot = 13.3E3
    g = 9.81
    t = 0.0
    def f(t): return u * math.log(M0 / (M0 - mdot * t)) - g * t

    def f_1(f, t, h=10E-4):
        return (f(t + h) - f(t - h)) / (2 * h)

    for i in range(30):
        dx = -(f(t)- v1) / f_1(f, t)
        t = t + dx
        if abs(dx) < acc: return t

'''
    Part 4: Systems of non-linear equations
'''

'''
    We will solve problem 4.1.26 from the textbook:
        "
        The equation of a circle is: (x-a)**2 + (y-b)**2 = R**2
        where R is the radius and (a,b) are the coordinates of the center.
        Given the coordinates of three points p1, p2 and p3, find a, b
        and R such that the circle of center (a, b) and radius R passes
        by p1, p2 and p3.
        "
'''

def f_4_1_26(x_data, y_data, x):
    a = x[0]
    b = x[1]
    r = x[2]

    def f(x, y): return (x - a) ** 2 + (y - b) ** 2 - r ** 2

    result = array([1.0, 1.0, 1.0])
    for i in range(len(x_data)):
        x1 = x_data[i]
        y1 = y_data[i]
        result[i] = f(x1, y1)

    return result

def problem_4_1_26(x_data, y_data):
    def f(xvec):
        x = xvec[0]
        y = xvec[1]
        z = xvec[2]
        return array([
            (x_data[0] - x) ** 2 + (y_data[0] - y) ** 2 - z ** 2,
            (x_data[1] - x) ** 2 + (y_data[1] - y) ** 2 - z ** 2,
            (x_data[2] - x) ** 2 + (y_data[2] - y) ** 2 - z ** 2
        ])

    x = array([1.0, 1.0, 1.0])
    solution = newtonRaphson2(f, x)
    return solution

'''
    Part 5: Interpolation and Numerical differentiation
'''

'''
    We will solve problem 5.1.11 from the textbook:
        " 1. Use polynomial interpolation to compute f' and f'' at x using
          the data in x_data and y_data:
                x_data = array([-2.2, -0.3, 0.8, 1.9])
                y_data = array([15.180, 10.962, 1.920, -2.040])
          2. Given that f(x) = x**3 - 0.3*x**2 -
             8.56*x + 8.448, gauge the accuracy of the result."
'''

def interpolant_5_1_11():
    return fit_poly([(-2.2, 15.180), (-0.3, 10.962), (0.8, 1.920), (1.9, -2.040)])

def d_dd_5_1_11(x):
    def f(x): return x ** 3 - 0.3 * x ** 2 - 8.56 * x + 8.448

    def f_1(f, x, h=10E-4):
        return (f(x + h) - f(x - h)) / (2 * h)

    def f_2(f, x, h=10E-4):
        return (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2

    return f_1(f, x), f_2(f, x)

def error_5_1_11(x):

    def f(x): return x ** 3 - 0.3 * x ** 2 - 8.56 * x + 8.448

    def f_1(f, x, h=10E-4): return (f(x + h) - f(x - h)) / (2 * h)

    def f_2(f, x, h=10E-4): return (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2

    return round(f_1(f, x), 2) - (-8.56), round(f_2(f, x), 2) - (-0.6)


def swapRows(v,i,j):
    if len(v.shape) == 1:
        v[i],v[j] = v[j],v[i]
    else:
        v[[i,j],:] = v[[j,i],:]

def gaussPivot(a,b,tol=1.0e-12):
    n = len(b)
    # Set up scale factors
    s = np.zeros(n)
    for i in range(n):
        s[i] = max(np.abs(a[i,:]))
    for k in range(0,n-1):
        # Row interchange, if needed
        p = np.argmax(np.abs(a[k:n,k])/s[k:n]) + k
        if p != k:
            swapRows(b,k,p)
            swapRows(s,k,p)
            swapRows(a,k,p)
# Elimination
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                b[i] = b[i] - lam*b[k]
    b[n-1] = b[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    return b

def newtonRaphson2(f,x,tol=1.0e-9):
    def jacobian(f,x):
        h = 1.0e-4
        n = len(x)
        jac = np.zeros((n,n))
        f0 = f(x)
        for i in range(n):
            temp = x[i]
            x[i] = temp + h
            f1 = f(x)
            x[i] = temp
            jac[:,i] = (f1 - f0)/h
        return jac,f0
    for i in range(30):
        jac,f0 = jacobian(f,x)
        if math.sqrt(np.dot(f0,f0)/len(x)) < tol: return x
        dx = gaussPivot(jac,-f0)
        x = x + dx
        if math.sqrt(np.dot(dx,dx)) < tol*max(max(abs(x)),1.0):
            return x

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
