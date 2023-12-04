import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

import graphviz

data = pd.read_csv('../Data/training_data_engineer_with_is_peak_hour.csv')

X = data.drop('increase_stock', axis = 1)
y = data['increase_stock']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 42)

tbm_model = tree.DecisionTreeClassifier(max_depth = 2)
tbm_model.fit(X_train, y_train)

print(tbm_model)

dot_data = tree.export_graphviz(tbm_model, out_file = None, feature_names = X_train.columns,
                                class_names = tbm_model.classes_, filled = True, rounded = True,
                                leaves_parallel = True, proportion = True)

graph = graphviz.Source(dot_data)
graph