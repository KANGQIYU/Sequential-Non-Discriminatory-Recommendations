# from sympy import Symbol, solve, log
# import numpy as np
# n = 4
# x = Symbol('x')
# a = np.random.rand(n-1,1)
# b = float(solve(x*log(x) - (3-a.T@np.log(a)), x)[0])
# print(b)

import cvxpy as cp
import numpy as np
import time
import dccp
from numpy import linalg as LA
w, v = LA.eig(np.diag((1, 2, 3)))
import matplotlib.pyplot as plt

# Problem data.
n = 4

# gamma must be nonnegative due to DCP rules.
c = cp.Parameter((n, 1))
ucbweight = cp.Parameter(nonneg=True)
A = np.eye(n)
# Construct the problem.
x = cp.Variable((n, 1))
w, v = LA.eigh(A)
sqrtw = np.sqrt(abs(w)).reshape(-1,1)
obj = cp.Maximize(c.T@x+ucbweight*cp.norm((cp.multiply(sqrtw, v*x))))

x.value = 1/np.ones((n, 1))
# constraints = [x.T@cp.log(x)<=5, x>=0]
constraints = [cp.sum(-cp.entr(x))<=5, x>=0]
# constraints = [cp.log_sum_exp(x)<=10, x>=0]

prob = cp.Problem(obj, constraints)
start_time = time.time()
# your code

# Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
for i in range(1):
    # c.value = np.random.rand(n, 1)
    ucbweight.value = np.random.rand(1)[0]
    c.value = np.asarray([[1., 1., 1., 1.]]).T

    # print("problem is DCP:", prob.is_dcp())  # false
    # print("problem is DCCP:", dccp.is_dccp(prob))  # true
    prob.solve(method='dccp')
    # Use expr.value to get the numerical value of
    # an expression in the problem.
    print("optimal var:\n", x.value)

elapsed_time = time.time() - start_time
print(elapsed_time)
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.figure(figsize=(6,10))
#
# # Plot trade-off curve.
# plt.subplot(211)
# plt.plot(l1_penalty, sq_penalty)
# plt.xlabel(r'\|x\|_1', fontsize=16)
# plt.ylabel(r'\|Ax-b\|^2', fontsize=16)
# plt.title('Trade-Off Curve for LASSO', fontsize=16)
#
# # Plot entries of x vs. gamma.
# plt.subplot(212)
# for i in range(m):
#     plt.plot(gamma_vals, [xi[i] for xi in x_values])
# plt.xlabel(r'\gamma', fontsize=16)
# plt.ylabel(r'x_{i}', fontsize=16)
# plt.xscale('log')
# plt.title(r'\text{Entries of x vs. }\gamma', fontsize=16)
#
# plt.tight_layout()
# plt.show()