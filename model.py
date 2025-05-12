import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import os
import xlrd
import cv2
import HandDataCollecter
import mediapipe as mp
import numpy as np

########Initialise random forest

local_path = (os.path.dirname(os.path.realpath('__file__')))

file_name = ('data.csv')  # file of total data
data_path = os.path.join(local_path, file_name)
print(data_path)
df = pd.read_csv(r'' + data_path)

print(df)

units_in_data = 28  # no. of units in data

titles = []
for i in range(units_in_data):
    titles.append("unit-" + str(i))
X = df[titles]
y = df['letter']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("***************************1.Random Forest Accuracy********************************")
clf = RandomForestClassifier(n_estimators=2000,    # Lowering the number of trees
    max_depth=5,        # Restricting the depth of each tree
    min_samples_split=5, # Increasing the minimum samples required to split
    min_samples_leaf=2)  # random forest
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
cmrf = confusion_matrix(y_test, y_pred)


print("Random Forest classification_report")
print(classification_report(y_pred, y_test, labels=None))
print("Random Forest confusion_matrix")
#print(cmrf)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("CONFUSION MATRIX OF RF")
print(cm)
tpr = float(cm[0][0]) / np.sum(cm[0])
fpr = float(cm[1][1]) / np.sum(cm[1])
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix of RF '
plt.title(all_sample_title, size=15);
plt.show()
print("**************************2.knn Accuracy******************************")
clf1 = KNeighborsClassifier(n_neighbors=50,       # Number of neighbors to consider
    weights='uniform',   # Weight function used in prediction
    leaf_size=30,        # Leaf size passed to BallTree or KDTree
    p=1)  # random forest
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
cmrf = confusion_matrix(y_test, y_pred)


print("knn classification_report")
print(classification_report(y_pred, y_test, labels=None))


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("CONFUSION MATRIX OF knn")
print(cm)
tpr = float(cm[0][0]) / np.sum(cm[0])
fpr = float(cm[1][1]) / np.sum(cm[1])
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix of knn '
plt.title(all_sample_title, size=15);
plt.show()
print("*****************3.svm Accuracy****************************")
from sklearn.svm import SVC
clf2 = SVC()  # random forest
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
cmsvc = confusion_matrix(y_test, y_pred)


print("svm classification_report")
print(classification_report(y_pred, y_test, labels=None))
print("svm confusion_matrix")
print(cmrf)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

tpr = float(cm[0][0]) / np.sum(cm[0])
fpr = float(cm[1][1]) / np.sum(cm[1])
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Confusion Matrix of svm '
plt.title(all_sample_title, size=15);
plt.show()

# Initialize individual models
clf_rf = RandomForestClassifier(n_estimators=100,  max_depth=None, )
clf_knn = KNeighborsClassifier()
clf_svc = SVC()
print("*****************4.Hybrid Stacking Model Accuracy****************************")
# Create a Stacking Classifier with a meta-model (Logistic Regression)
stacking_clf = StackingClassifier(estimators=[
    ('random_forest', clf_rf),
    ('knn', clf_knn),
    ('svm', clf_svc)
], final_estimator=LogisticRegression())

# Fit the stacking model on the training data
stacking_clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred_stacking = stacking_clf.predict(X_test)

# Evaluate the performance of the Stacking Classifier
stacking_accuracy = accuracy_score(y_test, y_pred_stacking)
print('Stacking Classifier Accuracy: {:.2%}'.format(stacking_accuracy))

print("Confusion Matrix of Stacking Classifier")
cm_stacking = confusion_matrix(y_test, y_pred_stacking)
print(cm_stacking)
import matplotlib.pyplot as plt

# Accuracy values
accuracies = [97.8, 94.4, 96.8, 98.15]
model_names = ['Random Forest', 'KNN', 'SVM', 'Hybrid']

# Plotting the bar graph
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color=['green', 'green', 'green', 'green'])
plt.title('Accuracy of Different Models')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)  # Set y-axis limit to represent accuracy percentage
plt.show()