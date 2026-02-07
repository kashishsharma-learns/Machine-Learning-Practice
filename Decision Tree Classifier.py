from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

iris=load_iris()
x,y=iris.data,iris.target

clf=DecisionTreeClassifier(criterion="entropy",random_state=0)
clf.fit(x,y)

tree_rules=export_text(clf,feature_names=iris['feature_names'])
print(tree_rules)

sample=[[5.1,3.5,1.4,0.2]]
prediction=clf.predict(sample)
print("Predicted class:",iris.target_names[prediction][0])