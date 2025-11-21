import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score

housing=datasets.fetch_california_housing()
x=housing.data
y=housing.target

x=x[:,[2]]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

svr=SVR(kernel="rbf",C=100,gamma=0.1,epsilon=0.1)
svr.fit(x_train,y_train)

y_pred=svr.predict(x_test)

print("MSE:",mean_squared_error(y_test,y_pred))
print("R2 Score:",r2_score(y_test,y_pred))

plt.scatter(x_test,y_test,color="darkorange",label="Actual")
plt.scatter(x_test,y_pred,color="navy",label="Predicted",alpha=0.6)
plt.xlabel("Average rooms per household")
plt.ylabel("House value")
plt.title("Support Vector Regression on California Housing")
plt.legend()
plt.show()