from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris=datasets.load_iris()

x=iris["data"][:,3:]

y=(iris["target"]==2).astype(int)

clf=LogisticRegression()
clf.fit(x,y)

example=clf.predict([[2.6]])
print("Prediction for 2.6:",example)

x_new=np.linspace(0,3,1000).reshape(-1,1)
y_prob=clf.predict_proba(x_new)

plt.figure(figsize=(8,5))
plt.scatter(x,y,c=y,cmap="bwr",edgecolors="k",label="Data points")
plt.plot(x_new,y_prob[:,1],"g-",linewidth=2,label="Probability of virginica")
plt.xlabel("Petal Width(cm)")
plt.ylabel("virginica (1=yes,0=no)")
plt.title("Logistic Regression on Iris Petal Width")
plt.legend()

plt.show()
