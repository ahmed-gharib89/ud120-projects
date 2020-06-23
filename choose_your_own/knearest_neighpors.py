#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()



################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from time import time

start = time()

clf = KNeighborsClassifier(n_neighbors=12, leaf_size=20, algorithm='ball_tree')
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(labels_test, pred)

print("acc is: {:.4f}".format(acc))
print("in: {:.2f} Seconds".format(time() - start))

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
