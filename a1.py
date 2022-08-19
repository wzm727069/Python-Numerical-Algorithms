from numpy import *
import numpy as np

def example_func():
    '''
      Important: READ THIS CAREFULLY. 
      Task: This function is an example, you don't have to modify it.
      Example: Nothing to report here, really.
      Test: This function is is tested in tests/test_example.py
            This test just gives you a bonus, yay!
      Hint: The functions below have to be implemented in Python, without
            using any function from numpy's linear algebra module. In each function, a
            docstring formatted as the present one explains what the 
            function must do (Task), gives an example of output 
            (Example), explains how it will be evaluated (Test), and 
            may give you some hints (Hint).
    '''
    return 'It works!'


def square(a):
    b = shape(a)
    if b[0] == b[1]:
        return True
    else:
        return False


'''
  Part 2: Resolution of linear systems for polynomial interpolation
'''


def fit_poly_2(points):
    x1 = points[0][0]
    x2 = points[1][0]
    x3 = points[2][0]
    A = [[x1 ** 2, x1, 1], [x2 ** 2, x2, 1], [x3 ** 2, x3, 1]];
    assert (isSingular(A, N))
    b = [-1, -2, -9];
    x = gauss(A, b);
    n = len(A);
    c = array([0, 0, 0])
    for i in range(0, n):
        c[i] = x[i]
    return c[::-1]


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


'''
  Part 3: Tridiagonal systems
'''
def tridiag_solver_n(n):
    a = []
    b = []
    c = []
    d = [9., 5.]
    for x in range(0, n):
        b.append(4.)
    for y in range(0, n - 1):
        a.append(-1.)
        c.append(-1.)
    for z in range(2, n):
        d.append(5.)
    nf = len(d)
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


'''
  Part 4: Gauss Elimination for more than 1 equation
'''
def gauss_multiple(a, b):
    n = len(b)
    for k in range(0, n - 1):
        for i in range(k + 1, n):
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                a[i, k + 1:n] = a[i, k + 1:n] - lam * a[k, k + 1:n]
                b[i] = b[i] - lam * b[k]

    for k in range(n - 1, -1, -1):
        b[k] = (b[k] - np.dot(a[k, k + 1:n], b[k + 1:n])) / a[k, k]
    return b

def gauss_multiple_pivot(a, b):

    tol = 1.0e-12
    n = len(b)

    s = zeros(n)
    for i in range(n):
        s[i] = max(abs(a[i, :]))

    for k in range(0, n - 1):

        p = argmax(abs(a[k:n, k]) / s[k:n]) + k
        if abs(a[p, k]) < tol: print('Matrix is singular')
        if p != k:
            swapRows(b, k, p)
            swapRows(s, k, p)
            swapRows(a, k, p)

        for i in range(k + 1, n):
            if a[i, k] != 0.0:
                lam = a[i, k] / a[k, k]
                a[i, k + 1:n] = a[i, k + 1:n] - lam * a[k, k + 1:n]
                b[i] = b[i] - lam * b[k]
    if abs(a[n - 1, n - 1]) < tol: print('Matrix is singular')

    b[n - 1] = b[n - 1] / a[n - 1, n - 1]
    for k in range(n - 2, -1, -1):
        b[k] = (b[k] - dot(a[k, k + 1:n], b[k + 1:n])) / a[k, k]
    return b


def matrix_invert(a):
    matrixinv = np.matrix(a)
    return matrixinv.I


global N
N = 3

def getCofactor(mat, temp, p, q, n):
    i = 0
    j = 0
    for row in range(n):
        for col in range(n):
            if (row != p and col != q):
                temp[i][j] = mat[row][col]
                j += 1

                if (j == n - 1):
                    j = 0
                    i += 1

def isSingular(mat, n):
    D = 0
    if (n == 1):
        return mat[0][0]

    temp = [[0 for i in range(N + 1)] for i in range(N + 1)]
    sign = 1
    for f in range(n):
        getCofactor(mat, temp, 0, f, n)
        D += sign * mat[0][f] * isSingular(temp, n - 1)

        sign = -sign
    return D

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

def swapRows(v, i, j):
    if len(v.shape) == 1:
        v[i], v[j] = v[j], v[i]
    else:
        temp = v[i].copy()
        v[i] = v[j]
        v[j] = temp


def swapCols(v, i, j):
    temp = v[:, j].copy()
    v[:, j] = v[:, i]
    v[:, i] = temp