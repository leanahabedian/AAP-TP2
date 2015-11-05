#!/usr/bin/python

from PIL import Image
import sys, os, math
from scipy.spatial.distance import euclidean
from sklearn import cluster, ensemble
from sklearn.neighbors import kneighbors_graph
import numpy as np
import shutil, os
import cv2 

try:
    shutil.rmtree("out")
except:
    pass
os.mkdir("out")


def armar_vector_color(filename):
    COLORS=256    # cantidad de colores para cuantizar las imagenes.
    im = Image.open(filename)
    im = im.convert("P", palette=Image.ADAPTIVE, colors=COLORS)
    cantpixels = float(im.size[0] * im.size[1])
    res = [0.0 for i in range(COLORS)]
    for (count, color) in im.getcolors():
        res[color] = count/cantpixels
    return res 

def armar_vector_gris(filename):
    img = cv2.imread(filename,0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ)) #stacking images side-by-side
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    histequ,binsequ = np.histogram(equ.flatten(),256,[0,256])
    return histequ


# Construye un collage de imagenes: una grilla cuadrada de thumbnails.
def armar_collage(images):
  THUMBSIZE=64  # ancho/alto en pixels de cada thumbnail.
  gridsize = int(math.ceil(math.sqrt(len(images))))  # ancho/alto (en imagenes) de la grilla
  res = Image.new("RGB", (THUMBSIZE*gridsize, THUMBSIZE*gridsize), color=0xBBBBBB)
  (x,y)= (0,0)
  for i in range(len(images)):
    thumbnail = Image.open(images[i])
    thumbnail.thumbnail((THUMBSIZE, THUMBSIZE))
    res.paste(thumbnail, (x*THUMBSIZE, y*THUMBSIZE))
    x += 1
    if x==gridsize:
      x=0
      y+=1
  return res

##############################################################

# Armo el listado de imagenes disponibles.
IMGDB = 'train'
NUMIMAGES = 100
filenames = ['%s/cat.%s.jpg' % (IMGDB, str(i))  for i in range(NUMIMAGES)]
filenames = filenames + ['%s/dog.%s.jpg' % (IMGDB, str(i))  for i in range(NUMIMAGES)]

# Extraigo el vector de atributos de cada imagen, y lo guardo en
# un diccionario data[filename] --> vector_atributos
data = {}
for filename in filenames:
  data[filename] = armar_vector_color(filename)

# Preparo la matriz con los datos de entrenamiento.
X = [data[filename] for filename in filenames]

#TODO: ELEGIR ACA UN ALGORITMO...
# Ejecuto el algoritmo de clustering.
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
#clustering = cluster.KMeans(n_clusters=2)
#clustering = cluster.DBSCAN()
#clustering = cluster.AgglomerativeClustering(linkage="average",connectivity=knn_graph,n_clusters=2)
#y = clustering.fit_predict(X)


classifier = ensemble.RandomForestClassifier()
X = classifier.fit(X, X).transform(X)
print classifier.predict_proba(X)

# Para cada imagen, averiguo el cluster al cual pertenece, 
# para incluir en el collage correspondiente.
numclusters = len(set(y))
clusters = [[] for i in range(numclusters)]
for i in range(len(filenames)):
  filename = filenames[i]
  cluster_index = y[i]
  clusters[cluster_index].append(filename)

# Dibujo el collage de cada cluster.
#TODO: ACTUALIZAR...
OUTDIR="./out/"
for i in range(numclusters):
  images = []
  cant_dogs = 0
  cant_cats = 0
  for filename in clusters[i]:
    images.append(filename)
    if 'cat' in filename: 
        cant_cats+=1
    else:
        cant_dogs+=1

  print ('cluster'+ str(i) +"cats:"+str(cant_cats) +" "+ "dogs:"+str(cant_dogs))
  
  collage = armar_collage(images)
  collage.save('%s/cluster-%02i.jpg' % (OUTDIR, i))



