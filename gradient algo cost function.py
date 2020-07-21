import numpy as np

def grad(x, y):
    mc = bc = 0
    iter = 10000
    n = len(x)
    learning_rate = 0.0001

    for i in range(iter):
        yp = mc * x + bc  ##y predicted
        cost=(1/n)*sum([val**2 for val in (y-yp)])
        md = -(2 / n) * sum(x * (y - yp))  ## M derivative
        bd = -(2 / n) * sum((y - yp))  ## B derivative
        mc = mc - learning_rate * md  ## M current
        bc = bc - learning_rate * bd
        print(" m {} ,b {},iteration {},cost {} ".format(mc, bc, i,cost))



x = np.array([1, 2, 3, 4, 5])
y = np.array([20, 30, 40, 50, 60])

grad(x,y)
