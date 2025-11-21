from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris=datasets.load_iris()
x=iris.data[:,:2]
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

clf=SVC(kernel='linear')
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
print("Accuracy:",accuracy_score(y_test,y_pred))

plt.figure(figsize=(8,6))
plt.scatter(x[:,0],x[:,1],c=y,cmap='viridis')
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("SVM on iris(linear kernel)")
plt.show()