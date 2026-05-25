# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and convert categorical labels into numerical form. 

2.Select input features and target output from the dataset. 

3.Normalize the feature values using StandardScaler. 

4.Initialize weights and define sigmoid and cost functions. 

5.Train the model using Gradient Descent to update weights iteratively. 

6.Predict output values, calculate accuracy, and plot cost vs iterations graph.

## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Hemapriya P
RegisterNumber:  212225040126

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SGDClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("Predicted Species:", iris.target_names[prediction][0])

plt.scatter(X[:,0], X[:,1], c=y)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Flower Classification")

plt.show()
```

## Output:

<img width="1205" height="721" alt="image" src="https://github.com/user-attachments/assets/73063889-24ab-4dca-a5fd-1729ba1b281d" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

