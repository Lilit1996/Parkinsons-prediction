#Importing necessary libraries

import numpy as np
import pandas as pd
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

#Plotting the correlation heatmap
corr = df.corr()
plt.figure(figsize=(15,8))
ax = sns.heatmap(corr, annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Correlation heatmap')
plt.show()

#Performing train test split of the data stratifying on y, as the target variable is imbalanced.
X = df.drop(['status','name'], axis = 1)
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = Config.test_size,
                                                    random_state = 42, stratify = y)

#Performing feature scaling
norm = StandardScaler().fit(X_train)
X_train_t = pd.DataFrame(columns = X_train.columns, data = norm.transform(X_train))
X_test_t = pd.DataFrame(columns = X_test.columns, data = norm.transform(X_test))

logreg = LogisticRegression(random_state = 8)
logreg_parameters = {'penalty': ['l1', 'l2'], 'C': np.logspace(-4, 4, 20),'class_weight':['balanced',None]}
grid_search = GridSearchCV(logreg, param_grid=logreg_parameters, cv = 5, scoring = Config.scoring)
grid_search.fit(X_train, y_train)
model_logreg = grid_search.best_estimator_
y_pred = grid_search.best_estimator_.predict(X_test)
y_train_pred = grid_search.best_estimator_.predict(X_train)

importance = model_logreg.coef_[0]
for i,v in enumerate(importance):
    print('Feature: %d, Score: %.4f' % (i,v))
plt.bar([col for col in df.columns.difference(["status", 'name'])], importance)
plt.xticks(df.columns.difference(["status", "name"]), rotation = '85')
plt.title('Logistic Regression feature importance')
plt.show()

tree = DecisionTreeClassifier(random_state = 8)
tree_parameters = {'max_depth': range(2, 10),'min_samples_split':range(2, 12, 2),'criterion':['gini','entropy'],'class_weight':['balanced',None]}
grid_search = GridSearchCV(tree, param_grid=tree_parameters, cv = 5, scoring = Config.scoring)
grid_search.fit(X_train, y_train)
model_tree = grid_search.best_estimator_
y_pred = grid_search.best_estimator_.predict(X_test)
y_train_pred = grid_search.best_estimator_.predict(X_train)

importance = model_tree.feature_importances_
for i,v in enumerate(importance):
    print('Feature: %d, Score: %.4f' % (i,v))
plt.bar([col for col in df.columns.difference(['status','name'])], importance)
plt.xticks(df.columns.difference(['status','name']), rotation = '85')
plt.title('Decision tree feature importance')
plt.show()

forest = RandomForestClassifier(random_state = 8)
forest_parameters = {'bootstrap': [True, False],
 'max_depth': range(8, 15),
 'min_samples_leaf': [1, 2, 4],
 'n_estimators': range(5, 20),
 'class_weight':['balanced', None]}
grid_search = GridSearchCV(forest, param_grid=forest_parameters, cv = 5, scoring = Config.scoring)
grid_search.fit(X_train, y_train)
model_forest = grid_search.best_estimator_
y_pred = grid_search.best_estimator_.predict(X_test)
y_train_pred = grid_search.best_estimator_.predict(X_train)

importance = model_forest.feature_importances_
for i,v in enumerate(importance):
    print('Feature: %d, Score: %.4f' % (i,v))
plt.bar([col for col in df.columns.difference(['status','name'])], importance)
plt.xticks(df.columns.difference(['status','name']), rotation = '85')
plt.title('Random forest feature importance')
plt.show()
