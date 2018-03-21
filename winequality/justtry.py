
# import pandas
# from sklearn import model_selection
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import ElasticNet
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.svm import SVR
# import numpy as np
# from sklearn.utils import check_random_state
# from sklearn.model_selection import train_test_split
#
# tmp = np.loadtxt("week-wti.csv", dtype=np.str, delimiter=",")
# X = tmp[1:, 0:8].astype(np.float)  # 加载数据部分
# Y = tmp[1:, 8].astype(np.float)  # 加载类别标签部分
#
# random_state = check_random_state(0)  # 将样本进行随机排列
# permutation = random_state.permutation(X.shape[0])
# X = X[permutation]
# Y = Y[permutation]
# X = X.reshape((X.shape[0], -1))
# train_x, test_x, train_y, test_y = train_test_split(
#         X, Y, test_size=0.2)
#
# def bulid_model(model_name):
#     model = model_name()
#     return model
# scoring1 = 'neg_mean_squared_error'
# scoring2 = 'explained_variance'
# scoring3 = 'r2'
#
# for model_name in [LinearRegression,Ridge,Lasso,ElasticNet,KNeighborsRegressor,DecisionTreeRegressor,SVR]:
#     model = bulid_model(model_name)
#     model.fit(train_x,train_y)
#     # results1 = model_selection.cross_val_score(model, X, Y, cv=10, scoring=scoring1)
#     # results2 = model_selection.cross_val_score(model, X, Y, cv=10, scoring=scoring2)
#     results3 = model_selection.cross_val_score(model, X, Y, cv=10, scoring=scoring3)
#     # print(model.score(train_x,train_y))
#     print(model.score(test_x, test_y))
#     # print(results3.mean())



import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

tmp = np.loadtxt("week-wti.csv", dtype=np.str, delimiter=",")
X = tmp[1:, 0:8].astype(np.float)  # 加载数据部分
Y = tmp[1:, 8].astype(np.float)  # 加载类别标签部分

random_state = check_random_state(0)  # 将样本进行随机排列
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
Y = Y[permutation]
X = X.reshape((X.shape[0], -1))

train_x, test_x, train_y, test_y = train_test_split(
    X, Y, test_size=0.2)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(train_x, train_y)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(test_x)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(test_y, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(test_y, diabetes_y_pred))