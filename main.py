# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#K(ro(u, xi)/hi) i = {1, ..., l}
u = x_train[1]
l = x_train.shape[0]

#%%

from numpy.linalg import norm
l2_u = norm(u, 2)
l2_x_train = norm(x_train, ord=2, axis=1)

#%%

ro = l2_u+l2_x_train
h = 1
a = 1
K = 1/(a+ro/h)

#%%

gamma = np.ones(K.size)

#%%

sum_y = (np.where(y_train == y_train[45], 1, 0)*gamma*K).sum()

#%%

y_ll = np.repeat(y_train, y_train.size, axis=0).reshape((y_train.size, y_train.size))
sum_y_ll = (np.where(y_ll == y_train, 1, 0)*gamma*K).sum(axis = 0)
sum_y_ll

#%%

y_train[sum_y_ll.argmax()]

#%%

train_size = y_train.size
l2_u = np.repeat(norm(x_train, ord=2, axis=1),2)
l2_u.shape

#%%
l2_x_train = (norm(x_train, ord=2, axis=1)).reshape( 1, train_size)
l2_x_train.repeat(l2_u.size, axis=0)
#%%
l2_x_train = np.repeat(norm(x_train, ord=2, axis=1), l2_u.size, axis=0)#.reshape(( l2_u.size, train_size))
#ro = l2_u+l2_x_train
l2_x_train

#%%

h = 1
a = 1
K = 1/(a+ro/h)
y_ll = np.repeat(y_train, y_train.size, axis=0).reshape((y_train.size, y_train.size))
sum_y_ll = (np.where(y_ll == y_train, 1, 0)*gamma*K).sum(axis = 0)

#%%

def a_what_class (u, x_train, y_train, gamma):
    l2_u = norm(u, 2)
    l2_x_train = norm(x_train, ord=2, axis=1)
    ro = l2_u+l2_x_train
    h = 1
    a = 1
    K = 1/(a+ro/h)
    y_ll = np.repeat(y_train.reshape( y_train.size, 1), y_train.size, axis=0).reshape((y_train.size, y_train.size))
    sum_y_ll = (np.where(y_ll == y_train, 1, 0)*gamma*K).sum(axis = 0)
    return y_train[sum_y_ll.argmax()]

#%%

def a_what_class (u, x_train, y_train, gamma):
    l2_u = norm(u, 2)
    l2_x_train = norm(x_train, ord=2, axis=1)
    ro = norm(x_train - u, ord=2, axis=1)
    h = 1
    a = 1
    K = 1/(a+ro/h)
    y_ll = np.repeat(y_train.reshape( y_train.size, 1), y_train.size, axis=1).transpose()
    sum_y_ll = (np.where(y_ll == y_train, 1, 0)*gamma*K).sum(axis = 0)
    return y_train[sum_y_ll.argmax()]
