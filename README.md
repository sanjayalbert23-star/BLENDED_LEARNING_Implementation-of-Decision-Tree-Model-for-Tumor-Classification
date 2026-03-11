# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries such as Pandas,Numpy,and Scikit learn
2. Preprocess the data
3. Seperate the dataset into Input features and Target variable 
4. Split the dataset into training data and testing data using train-test split
5. Create the decision tree classifier model.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('tumor.csv')

print(data.head())
print(data.columns)

X = data.drop(columns=['Class'])  
y = data['Class'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nName: SANJAY A")
print("Reg No: 212225040367")
print("\nAccuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))


conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: SANJAY A
RegisterNumber:  212225040367
*/
```

## Output:
<img width="867" height="360" alt="Screenshot 2026-03-11 131830" src="https://github.com/user-attachments/assets/1b35b1a6-3ee2-4e84-9c31-2255fe189504" />
<img width="984" height="879" alt="Screenshot 2026-03-11 132032" src="https://github.com/user-attachments/assets/70d68aeb-0a6c-494b-b1c0-e1e4ef8edc70" />



## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
