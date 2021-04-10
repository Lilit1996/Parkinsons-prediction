import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
sns.set_palette('YlOrRd_r', n_colors = 3)
import sys
sys.path.append('../../configs')
import Config
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../data/parkinsons_data.txt")
X = df.drop(['status','name','MDVP:APQ','MDVP:Jitter(%)','MDVP:PPQ','MDVP:RAP'],axis = 1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = Config.test_size,
                                                    random_state = 42, stratify = y)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

norm = StandardScaler().fit(X_train)
X_train_t = pd.DataFrame(columns = X_train.columns, data = norm.transform(X_train))
X_test_t = pd.DataFrame(columns = X_test.columns, data = norm.transform(X_test))

knn = KNeighborsClassifier(n_neighbors = int(math.sqrt(len(df))))
knn.fit(X_train_t, y_train)

y_train_pred = knn.predict(X_train_t)
y_pred = knn.predict(X_test_t)

print('Model evaluation for training data')
print("Accuracy score")
print(accuracy_score(y_train, y_train_pred))
print('\n')
print("Classification report")
print(classification_report(y_train, y_train_pred))
print('\n')
print("ROC AUC score")
print(roc_auc_score(y_train, y_train_pred))

cf_matrix = confusion_matrix(y_train, y_train_pred)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Confusion matrix of train data')
plt.show()

print('Model evaluation for test data')
print("Accuracy score")
print(accuracy_score(y_test, y_pred))
print('\n')
print("Classification report")
print(classification_report(y_test, y_pred))
print('\n')
print("ROC AUC score")
print(roc_auc_score(y_test, y_pred))

cf_matrix = confusion_matrix(y_test, y_pred)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Confusion matrix of test data')

plt.show()

scores = cross_val_score(knn, X_train_t, y_train, cv=10, scoring = Config.scoring)
print(scores)
print('\n')
print(scores.mean())


