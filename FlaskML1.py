# load the iris dataset as an example
from sklearn.datasets import load_iris

iris = load_iris()
# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target
#print(X)
#print(y)
# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)

print(y_train)

print(X_test)

print(y_test)

# training the model on training set
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# making predictions on the testing set
y_pred = knn.predict(X_test)
print(y_pred)
# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics

print("kNN model accuracy:", metrics.accuracy_score(y_test, y_pred))

import pickle
with open("knn.pkl",'wb') as model_pkl:
    pickle.dump(knn,model_pkl)

#It worked on 20-01-2020