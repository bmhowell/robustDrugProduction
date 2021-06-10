import cvxpy as cp
import numpy as np


def drugOpt(costs_):
    cost1 = costs_[0]
    cost2 = costs_[1]
    a1 = [0.1, 0.0095, 0.01005]
    a2 = [0.2, 0.0196, 0.0204]
    A = np.array([[-a1[1], -a2[1], 0.5, 0.6],
                  [1, 1, 0, 0],
                  [0, 0, 90.0, 100.0],
                  [0, 0, 40.0, 50.0],
                  [100.0, 199.9, cost1, cost2]])
    b = np.array([0, 1000, 2000, 800, 100000])
    c = np.array([100, 199.9, cost1-5500, cost2-6100])

    x = cp.Variable(4)

    obj = cp.Minimize(c.T @ x)

    constraints = [x >= 0,
                   A@x <= b]

    prob = cp.Problem(obj, constraints)
    prob.solve()
    optimalX = x.value
    profit = prob.value

    return optimalX, profit

# opCost = np.array([[0, 0]])
opCost = np.array([[0, 0],
                   [100,150],
                   [200, 275],
                   [400, 500],
                   [600, 675],
                   [700, 800]])

profitTot = []
for i in range(len(opCost)):
    optimalX_, profit_ = drugOpt(opCost[i, :])

    profitTot.append(profit_)


import matplotlib.pyplot as plt

xplot = [np.sum(opCost[i, :]) for i in range(len(opCost[:, 0]))]

print(xplot)
profitTot = [-1*profitTot[i] for i in range(len(profitTot))]
print('profitTot: ', profitTot)


plt.figure()
plt.plot(xplot, profitTot, '-o')
plt.plot([-100, 2000], [0, 0], '-r')
plt.ylabel('Profit ($)')
plt.xlabel('Total Operating Costs ($)')

# plt.show()
plt.savefig('drugOpt.png')


# """
# # Uncertainty
#
# a1 = [0.0095, 0.01005]
# a2 = [0.0196, 0.0204]
# +a1[0]
# +a2[0]
# """
# cost1 = 0
# cost2 = 0
# a1 = [0.0095, 0.01005]
# a2 = [0.0196, 0.0204]
# A = np.array([[-(a1[0]), -(a2[1]), 0.5, 0.6],
#                   [1, 1, 0, 0],
#                   [0, 0, 90.0, 100.0],
#                   [0, 0, 40.0, 50.0],
#                   [100.0, 199.9, 700, 800]])
# b = np.array([0, 1000, 2000, 800, 100000])
# c = np.array([100, 199.9, cost1-5500, cost2-6100])
#
# x = cp.Variable(4)
#
# obj = cp.Minimize(c.T @ x)
#
# constraints = [x >= 0,
#                A@x <= b]
#
# prob = cp.Problem(obj, constraints)
# prob.solve()
# robustX = x.value
# robustProfit = -prob.value
#
# print('robustX: ', np.round(robustX, 5))
# print('robustProfit: ', robustProfit)
# print('')

# # original module2 solution
#
# cost1 = 700
# cost2 = 800
# A = np.array([[-0.01, -0.02, 0.5, 0.6],
#               [1, 1, 0, 0],
#               [0, 0, 90.0, 100.0],
#               [0, 0, 40.0, 50.0],
#               [100.0, 199.9, cost1, cost2]])
# b = np.array([0, 1000, 2000, 800, 100000])
# c = np.array([100, 199.9, cost1 - 5500, cost2 - 6100])
#
# x = cp.Variable(4)
#
# obj = cp.Minimize(c.T @ x)
#
# constraints = [x >= 0,
#                A @ x <= b]
#
# prob = cp.Problem(obj, constraints)
# prob.solve()
# optimalX = x.value
# profit = prob.value
#
# print('optimalX: ', (optimalX))
# print('profit: ', profit)
