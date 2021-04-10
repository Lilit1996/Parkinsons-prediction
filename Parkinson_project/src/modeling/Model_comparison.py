#Importing necessary libraries
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
model_logreg = LogisticRegression(C=1.0, class_weight=None, dual=False,
                            fit_intercept=True,
                            intercept_scaling=1, l1_ratio=None,
                            max_iter=100, multi_class='auto',
                            n_jobs=None, penalty='l2',
                            random_state=8, solver='lbfgs',
                            tol=0.0001, verbose=0,
                            warm_start=False)
model_tree = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                    criterion='gini', max_depth=None,
                                    max_features=None,
                                    max_leaf_nodes=None,
                                    min_impurity_decrease=0.0,
                                    min_impurity_split=None,
                                    min_samples_leaf=1,
                                    min_samples_split=2,
                                    min_weight_fraction_leaf=0.0,
                                    presort='deprecated',
                                    random_state=8, splitter='best')
model_forest = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
class_weight=None,
criterion='gini', max_depth=None,
max_features='auto',
max_leaf_nodes=None,
max_samples=None,
min_impurity_decrease=0.0,
min_impurity_split=None,
min_samples_leaf=1,
min_samples_split=2,
min_weight_fraction_leaf=0.0,
n_estimators=100, n_jobs=None,
oob_score=False, random_state=8,
verbose=0, warm_start=False)

models = []
models.append(('KNN', knn))
models.append(('Logistic Regression', model_logreg))
models.append(('Decision Tree Classifier', model_tree))
models.append(('Random Forest', model_forest))
acc_results = []
auc_results = []
f_1_results = []
names = []

col = ['Algorithm', 'ROC AUC Mean', 'ROC AUC STD',
       'Accuracy Mean', 'Accuracy STD',"F1 score Mean", "F1 score STD"]
df_results = pd.DataFrame(columns=col)

i = 0
for name, model in models:
    cv_acc_results = cross_val_score(model, X_train_t, y_train, cv=5, scoring='accuracy')
    cv_auc_results = cross_val_score(model, X_train_t, y_train, cv=5, scoring='roc_auc')
    cv_f_1_results = cross_val_score(model, X_train_t, y_train, cv=5, scoring='f1')

    acc_results.append(cv_acc_results)
    auc_results.append(cv_auc_results)
    f_1_results.append(cv_f_1_results)
    names.append(name)
    df_results.loc[i] = [name,
                         round(cv_auc_results.mean()*100, 2),
                         round(cv_auc_results.std()*100, 2),
                         round(cv_acc_results.mean()*100, 2),
                         round(cv_acc_results.std()*100, 2),
                         round(cv_f_1_results.mean()*100, 2),
                         round(cv_f_1_results.std()*100, 2)
                         ]
    i += 1
df_results.sort_values(by=['Accuracy Mean'], ascending=False)

fig = plt.figure(figsize=(10, 5))
fig.suptitle('Algorithm Accuracy Comparison')
ax = fig.add_subplot(111)
plt.boxplot(acc_results)
ax.set_xticklabels(names)
plt.show()


fig = plt.figure(figsize=(10, 5))
fig.suptitle('Algorithm ROC AUC Comparison')
ax = fig.add_subplot(111)
plt.boxplot(auc_results)
ax.set_xticklabels(names)
plt.show()
