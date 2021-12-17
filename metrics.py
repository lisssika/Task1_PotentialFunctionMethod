import numpy as np
from numpy.linalg import norm

def a_what_class (u, x_train, y_train, gamma, classes):
    l2_u = norm(u, 2)
    l2_x_train = norm(x_train, ord=2, axis=1)
    ro = norm(x_train - u, ord=2, axis=1)
    h = 1
    a = 1
    K = 1/(a+ro/h)
    gamma_Kernel = gamma*K
    #y_ll = np.repeat(y_train.reshape( y_train.size, 1), classes.size, axis=1).transpose()
    #sum_y_ll = (np.where(y_ll == y_train, 1, 0)*gamma*K).sum(axis = 0)
    sum_by_class = np.ones(classes.size)
    j = 0
    for i in classes:
        sum = (np.where(y_train == i, 1, 0)*gamma_Kernel).sum()
        sum_by_class[j] = sum
        j = j+1
    return sum_by_class.argmax()
