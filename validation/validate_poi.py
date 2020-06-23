#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset2.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys2.pkl')
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

features_train, features_test, labeles_train, labeles_test = train_test_split(
                                                            features, labels, random_state = 42, test_size = 0.3)

dt = DecisionTreeClassifier()
dt.fit(features_train, labeles_train)
pred = dt.predict(features_test)
acc = accuracy_score(labeles_test, pred)

print(f'The Accurecy is: {acc}')
