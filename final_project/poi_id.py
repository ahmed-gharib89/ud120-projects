#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','deferral_payments', 'total_payments',
                'loan_advances', 'bonus', 'restricted_stock_deferred',
                'deferred_income', 'total_stock_value', 'expenses',
                'exercised_stock_options', 'other', 'long_term_incentive',
                'restricted_stock', 'director_fees', 'shared_receipt_with_poi',
                'to_messages','from_messages', 'from_poi_to_this_person',
                'from_this_person_to_poi'] # First I try with all features available

### Load the dictionary containing the dataset
with open("final_project_dataset2.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
print(f'Data Lenght: {len(data_dict)}')
### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
data_dict.pop('TOTAL', 0)
print(f'Data length after removing outliers: {len(data_dict)}')
keys = data_dict.keys()
df = pd.DataFrame.from_dict(data_dict, orient='index', columns=features_list).replace('NaN', np.nan)
print(df.head())
print(df.info())
print(df.describe())

# after seeing my data and missing values now I will manually remove features
# with lots of missing values
col_to_remove = [col for col in df.columns if df[col].isna().sum() >= 80]
print(len(col_to_remove))
print(col_to_remove)
print(f'No of Features: {len(features_list)}')
features_list = [feature for feature in features_list if feature not in col_to_remove]
print(features_list)
print(f'No of Features after removing features with lots of nulls: {len(features_list)}')


my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(min_samples_split=7)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc_score = accuracy_score(labels_test, pred)
dt_precision_score = precision_score(labels_test, pred)
dt_recall_score = recall_score(labels_test, pred)
dt_confusion_matrix = confusion_matrix(labels_test, pred)
print(f'Accuracy score with DT is: {acc_score}')
print(f'Precision score with DT is: {dt_precision_score}')
print(f'Recall score with DT is: {dt_recall_score}')
print(dt_confusion_matrix)

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from time import time

start = time()
estimators = [('reduce_dim', PCA()), ('clf', SVC())]
pipe = Pipeline(estimators)

param_grid = dict(reduce_dim__n_components=[3, 5, 8, 10], clf__C=[0.1, 10, 100], clf__kernel=['linear', 'rbf'])
#param_grid = dict(reduce_dim__n_components=[5, 10], clf__C=[0.1, 10, 100])
clf = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, scoring='f1', cv=5, return_train_score=True)
clf.fit(features_train, labels_train)
print(f'Grid Search score: {clf.score(features_test, labels_test)}')
print(sorted(clf.cv_results_.keys()))
print(f'Best params: {clf.best_params_}')
#print(clf.cv_results_)

print(f'Finished in: {time()-start:.2f} seconds')
#print(f'Grid Search precision: {precision_score(features_test, labels_test)}')
#print(f'Grid Search recall: {recall_score(features_test, labels_test)}')

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.model_selection import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
