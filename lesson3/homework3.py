# Q1 rewrite the linear regression without explicitly using for loops


import numpy as np


def para_grad(X, Y, predict, type='MSE'):
    # dw = dcoss/dpredict * dpredict/dw

    if type == "MSE":
        loss = predict - Y
        dw = 2 / len(Y) * np.dot(loss, X.T)

        # equal to:
        # for i in loss:
        #     dw += 2* len(Y) * loss[i] * X[i].T        # (1*1*1*3) sample loop = 2

        dw = dw.T
        db = 2 * np.sum(loss * 1) / len(Y)

        # print(
        #     "predict = %s "%(predict),
        #     "loss = %s"%(loss),
        # )

        para_grad = {"dW": dw, "db": db, 'loss': np.sum(loss)}


        return para_grad


def predict(X, W, b, acti_func=0):
    num_features, num_sample = X.shape
    assert num_features, 1 == W.shape()

    predict = np.dot(W.T, X) + b

    parameter = {"W": W, "b": b}

    if acti_func:
        predict = sigmoid(predict)
        print(predict)
    return predict, parameter


def data_update(grad, parameter, lr=0):
    parameter['W'] -= lr * grad['dW']
    parameter['b'] -= lr * grad['db']

    return parameter


def train(X_nplist, Y_nplist, lr, num_epoch=1000):
    num_features, num_sample = X_nplist.shape  # shape X = (3,2)
    W = np.zeros((num_features, 1))  # shape W = (3,1) W.T = (1,3)  W.T X = (1*3*3*2) = 1*2 = shape Y
    b = 0  # b = 0 as program in numpy can use broadcasting to each prediction

    paras = {}
    loss = 0

    for epoch in range(num_epoch):

        pred, parameter = predict(X_nplist, W, b)
        grad = para_grad(X_nplist, Y_nplist, pred)
        loss = grad['loss']
        if epoch % 5 == 0:
            pass
            print(
                "\n loss = %s at epoch %s" % (loss, epoch)
            )
        paras = data_update(grad, parameter, lr)

    # print("\n finish training")

    return (paras, loss)


def sigmoid(x):
    sigmoids = 1 / (np.exp(-1 * x) + 1)
    return sigmoids


# X = np.array([[1, 2, 3], [5, 6, 7]]).reshape(3, 2)
# Y = [1, 2]
#
# result = train(X, Y, 0.01)
# print(
#     "\nparameters:",
#     "\nw = {} ".format(result[0]["W"]),
#     "\nb = {} ".format(result[0]["b"]),
#     "\nloss = {}".format(result[1])
# )


# Q2 write a logistic regression function


def train_log(X_nplist, Y_nplist, lr, num_epoch=1000):
    num_features, num_sample = X_nplist.shape  # shape X = (3,2)
    W = np.zeros((num_features, 1))  # shape W = (3,1) W.T = (1,3)  W.T X = (1*3*3*2) = 1*2 = shape Y
    b = 0  # b = 0 as program in numpy can use broadcasting to each prediction

    paras = {}
    loss = 0

    for epoch in range(num_epoch):

        pred, parameter = predict(X_nplist, W, b, 1)
        grad = para_grad(X_nplist, Y_nplist, pred)
        loss = grad['loss']
        if epoch % 5 == 0:
            pass
            print(
                "\n loss = %s at epoch %s" % (loss, epoch)
            )
        paras = data_update(grad, parameter, lr)

    # print("\n finish training")

    return (paras, loss)


X = np.array([[1, 2, 3], [5, 6, 7]]).reshape(3, 2)
Y = [1, 0]

result = train_log(X, Y, 0.01)
prediction = predict(X,result[0]["W"],result[0]["b"],1)

print(
    "\nparameters:",
    "\nw = {} ".format(result[0]["W"]),
    "\nb = {} ".format(result[0]["b"]),
    "\nloss = {}".format(result[1]),
    "\n prediction = {}".format((prediction[0]>0.5).astype(int))
)


# Q3
#
# I think this question is a convex optimization problem
# solving by lagrange dual function
# the question could be re-described as following way:
#
# Max Ds = sum(ds1 to dsi)
#
# sub to:
#
# euqality equations:
#
# ds1 - t1(v1 + v cos(a1)) == 0
# ds2 - t2(v2 + v cos(a2)) == 0
# dsi - ti(vi + v cos(ai)) == 0
# t1* v * sin(a1) - s1 == 0
# t2* v * sin(a2) - s2 == 0
# ti* v * sin(ai) - si == 0
#
# ineuqality equations:
#
# sum(t1 to ti) - T <= 0
#
# after introducing the dual function and lagrange muiltipler:
# the question could be as:
#
# Dual function = sum(from ds1 to dsi) - lenda * \
#
# lenda * sum(ds1 - t1(v1 + v cos(a1)  t  dsi - ti(vi + v cos(ai)) +\
# + sum(t1* v * sin(a1) - s1 to ti* v * sin(ai) - si)) +
#
# mu * (sum(t1 to ti) - T)
# for i from 1 to number of rivers. subject to lenda and mu