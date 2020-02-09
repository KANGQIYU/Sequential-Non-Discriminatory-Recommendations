from cvxopt import solvers, matrix, spdiag, log, div, sqrt
import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt

# Problem data.

# def acent(A, b):
#     m, n = A.size
#     def F(x=None, z=None):
#         if x is None: return 0, matrix(1.0, (n,1))
#         if min(x) <= 0.0: return None
#         f = -sum(log(x))
#         Df = -(x**-1).T
#         if z is None: return f, Df
#         H = spdiag(z[0] * x**-2)
#         return f, Df, H
#     return solvers.cp(F, A=A, b=b)['x']
# >>> c = matrix([-6., -4., -5.])
# >>> G = matrix([[ 16., 7.,  24.,  -8.,   8.,  -1.,  0., -1.,  0.,  0.,
#                    7.,  -5.,   1.,  -5.,   1.,  -7.,   1.,   -7.,  -4.],
#                 [-14., 2.,   7., -13., -18.,   3.,  0.,  0., -1.,  0.,
#                    3.,  13.,  -6.,  13.,  12., -10.,  -6.,  -10., -28.],
#                 [  5., 0., -15.,  12.,  -6.,  17.,  0.,  0.,  0., -1.,
#                    9.,   6.,  -6.,   6.,  -7.,  -7.,  -6.,   -7., -11.]])
# >>> h = matrix( [ -3., 5.,  12.,  -2., -14., -13., 10.,  0.,  0.,  0.,
#                   68., -30., -19., -30.,  99.,  23., -19.,   23.,  10.] )
# >>> dims = {'l': 2, 'q': [4, 4], 's': [3]}
# >>> sol = solvers.conelp(c, G, h, dims)

c = np.asarray([[1., 1.]])
# n = c.shape[1]
n = 4
c= matrix(np.random.rand(n, 1))



I = matrix(0.0, (n,n))
I[::n+1] = 1.0
# G = matrix([-I, matrix(0.0, (1,n)), I])
# h = matrix(n*[0.0] + [5.0] + n*[0.0])
# dims = {'l': n, 'q': [n+1], 's': []}


G = -I
h = matrix(n*[0.0])
dims = {'l': n, 'q': [], 's': []}
solvers.options['show_progress'] = False


def lin_entropy_constraint(c,n):
    c_m = matrix(-c)
    m = 1
    def F(x=None, z=None):
        if x is None: return m, matrix(1.0/n, (n,1))
        if min(x)<=0: return None
        Df = log(x)+1
        val = sum(x.T * log(x))-5
        if z is None: return val, Df.T
        H = spdiag(z[0] * div(1, x))
        return val, Df.T, H
    return solvers.cpl(c_m, F, G, h, dims)['x']

def lin_entropy_constraint1(c,n):
    c_m = matrix(-c)
    I = matrix(0.0, (n, n))
    I[::+ 1] = 1.0
    m = 1
    G = matrix([matrix(0.0, (1, n)), I])
    h = matrix([1.0] + n * [0.0])
    dims = {'l': 0, 'q': [n+1], 's': []}
    def F(x=None, z=None):
        if x is None: return m, matrix(0.0, (n, 1))
        if max(abs(x)) >= 1.0: return None
        u = 1 - x ** 2
        val = -sum(log(u))-1
        Df = div(2 * x, u).T
        if z is None: return val, Df
        H = spdiag(2 * z[0] * div(1 + u ** 2, u ** 2))
        return val, Df, H
    return solvers.cpl(c_m, F, G, h, dims)['x']

# np.sqrt(np.sum(Arms * (V_t_inv_linucb @ Arms), axis=0)).
def ucb_entropy_constraint(A, constant, c, n):
    c_m = matrix(-c)
    A = matrix(A)
    constant = -constant
    m = 1
    def F(x=None, z=None):
        if x is None: return m, matrix(1.0/n, (n,1))
        if min(x)<=0: return None
        temp = sqrt(x.T * A * x)
        Df = matrix([ c_m+ constant/temp*A*x, log(x)+1])
        val = matrix([c_m.T* x + constant*temp, sum(x.T * log(x))-5])
        if z is None: return val.T, Df.T
        H = z[0]* (temp*A-A*x*x.T*A/temp)/(temp**2)+spdiag(z[1] * div(1, x))
        return val.T, Df.T, H.T
    return solvers.cp(F, G, h, dims)['x']

# start_time = time.time()
for i in range(1000):
    c= matrix(np.random.rand(n, 1))
    # c.value = np.asarray([[1., 1.]]).T
    lin_entropy_constraint1(c, n)
    # Use expr.value to get the numerical value of
    # an expression in the problem.
#     # print("optimal var:\n", x.value)
#
# elapsed_time = time.time() - start_time
# print(elapsed_time)

print(lin_entropy_constraint1(c,n))
# print(ucb_entropy_constraint(I, 1, c, n))

