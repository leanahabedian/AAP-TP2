#!/usr/bin/python

from sklearn import ensemble, cross_validation
import shutil, os
from ensemble_classifier import EnsembleClassifier
from armar_vector import ArmarVector


try:
    shutil.rmtree("out")
except:
    pass
os.mkdir("out")


# generate many random forest
def generate_many_random_forest(X_train, X_test, y_train, y_test, label, estimators, leafs=1):

  def generate_rf(X_train, y_train, X_test, y_test):
    rf = ensemble.RandomForestClassifier(n_estimators=estimators, min_samples_leaf=leafs)
    rf.fit(X_train, y_train)
    return rf

  def combine_rfs(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a


  rfs = [generate_rf(X_train, y_train, X_test, y_test) for i in xrange(10)]
  # in this step below, we combine the list of random forest models into one giant model
  rf_combined = reduce(combine_rfs, rfs)
  # the combined model scores better than *most* of the component models
  scores = cross_validation.cross_val_score(rf_combined, X, y, cv=5, scoring='accuracy') 
  print("Accuracy: %0.2f (+/- %0.2f) %s" % (scores.mean(), scores.std(), label))
  return rf_combined

# generate only one random forest
def generate_only_one_random_forest(X_train, X_test, y_train, y_test, label, estimators, leafs=1):

  classifier = ensemble.RandomForestClassifier(n_estimators=estimators, min_samples_leaf=leafs)
  classifier.fit(X_train,y_train)
  scores = cross_validation.cross_val_score(classifier, X, y, cv=5, scoring='accuracy') 
  print("Accuracy: %0.2f (+/- %0.2f) %s" % (scores.mean(), scores.std(), label))


# Armo el listado de imagenes disponibles.
IMGDB = 'train'
NUMIMAGES = 12500
filenames_cats = ['%s/cat.%s.jpg' % (IMGDB, str(i))  for i in range(NUMIMAGES)]
filenames_dogs = ['%s/dog.%s.jpg' % (IMGDB, str(i))  for i in range(NUMIMAGES)]
print"Amount of images per category:", NUMIMAGES

# Extraigo el vector de atributos de cada imagen, y lo guardo en
# un diccionario data[filename] --> vector_atributos
data = {}
y = []
# class 0 is cat, class 1 is dog
for filename in filenames_cats:

  av = ArmarVector(filename)
  data[filename] = av.armar_vector_gris()
  #data[filename] = av.armar_vector_color()
  #data[filename]  = av.armar_vector_dark_pix_2x2()
  #data[filename]  = av.armar_vector_claro_oscuro_5x5()
  #data[filename]  = av.armar_vector_claro_oscuro_5x5_sin_overlapping()
  #data[filename]  = av.armar_vector_color_5x5_sin_overlapping()
  #data[filename]  = av.armar_vector_mahotas()
  #data[filename]  = av.armar_vector_mix()
  y.append(0)

for filename in filenames_dogs:

  av = ArmarVector(filename)
  data[filename] = av.armar_vector_gris()
  #data[filename] = av.armar_vector_color()
  #data[filename]  = av.armar_vector_dark_pix_2x2()
  #data[filename]  = av.armar_vector_dark_pix_5x5_claro_oscuro()
  #data[filename]  = av.armar_vector_dark_pix_5x5_claro_oscuro_sin_overlapping()
  #data[filename]  = av.armar_vector_dark_pix_5x5_color_sin_overlapping()
  #data[filename]  = av.armar_vector_mahotas()
  #data[filename]  = av.armar_vector_mix()
  y.append(1)

# Preparo la matriz con los datos de entrenamiento.
X = [data[filename] for filename in filenames_dogs] + [data[filename] for filename in filenames_cats]

print "amount of attributes extracted",len(X[0])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
generate_many_random_forest(X_train, X_test, y_train, y_test, "[Random Forest Mixed]", 100)

train_initial = [row[0:26] for row in X_train]
test_initial = [row[0:26] for row in X_test]

train_last = [row[26:] for row in X_train]
test_last = [row[26:] for row in X_test]

clf1 = generate_many_random_forest(train_initial, test_initial, y_train, y_test, "[Random Forest 5x5 claro/oscuro]", 100)
clf2 = generate_many_random_forest(train_last, test_last, y_train, y_test, "[Random Forest Mahotas]", 1000)
eclf = EnsembleClassifier(clfs=[clf1, clf2], voting='soft', weights=[1,3])
scores = cross_validation.cross_val_score(eclf, X, y, cv=5, scoring='accuracy')

print("Accuracy: %0.2f (+/- %0.2f) %s" % (scores.mean(), scores.std(), "[Random Forest Weighted]"))
