import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
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

logreg = LogisticRegression(random_state = 8)
logreg_parameters = {'penalty': ['l1', 'l2'], 'C': np.logspace(-4, 4, 20),'class_weight':['balanced',None]}
grid_search = GridSearchCV(logreg, param_grid=logreg_parameters, cv = 5, scoring=Config.scoring)
grid_search.fit(X_train, y_train)

model_logreg = grid_search.best_estimator_

y_pred = grid_search.best_estimator_.predict(X_test)
y_train_pred = grid_search.best_estimator_.predict(X_train)

print('Model evaluation for train data')
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
plt.title('Confusion matrix of train data')
plt.show()

scores = cross_val_score(logreg, X_train_t, y_train, cv=10, scoring = Config.scoring)
print(scores)
print(scores.mean())

from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, grid_search.best_estimator_.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, grid_search.best_estimator_.predict_proba(X_test_t)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

