#!/usr/bin/python

from PIL import Image
import sys, os, math
from scipy.spatial.distance import euclidean
from sklearn import cluster, ensemble, cross_validation
from sklearn.neighbors import kneighbors_graph
import numpy as np
import shutil, os
import cv2 
from scipy.spatial import distance


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

def oscuro(color_pix):
  #return 1 if color_pix > 128 else 0
  return 1 if color_pix > 200 else 0

# adding the darkness of a 2x2 frame
def armar_vector_dark_pix_2x2(filename):
  res = [0 for i in xrange(0,5)]
  img = cv2.imread(filename,0)
  equ = cv2.equalizeHist(img)
  for i in range(equ.shape[0]-1):
    for j in range(equ.shape[1]-1):
      p = oscuro(equ[i,j]) + oscuro(equ[i+1,j]) + oscuro(equ[i+1,j]) + oscuro(equ[i+1,j+1])
      res[p] += 1
  return res

# adding the darkness of a 5x5 frame
def armar_vector_dark_pix_5x5_claro_oscuro(filename):
  res = [0 for i in xrange(0,26)]
  img = cv2.imread(filename,0)
  equ = cv2.equalizeHist(img)
  for i in range(equ.shape[0]-4):
    for j in range(equ.shape[1]-4):
      p = 0
      for k in xrange(0,5):
        p += oscuro(equ[i,j+k]) + oscuro(equ[i+1,j+k]) + oscuro(equ[i+2,j+k]) + oscuro(equ[i+3,j+k]) + oscuro(equ[i+4,j+k])
      res[p] += 1
  return res

# adding the darkness of a 5x5 frame
def armar_vector_dark_pix_5x5_claro_oscuro_sin_overlapping(filename):
  res = [0 for i in xrange(0,26)]
  img = cv2.imread(filename,0)
  equ = cv2.equalizeHist(img)
  dim = 5
  for i in range(0,equ.shape[0]-4,dim):
    for j in range(0,equ.shape[1]-4,dim):
      p = 0
      for k in xrange(0,5):
        p += oscuro(equ[i,j+k]) + oscuro(equ[i+1,j+k]) + oscuro(equ[i+2,j+k]) + oscuro(equ[i+3,j+k]) + oscuro(equ[i+4,j+k])
      res[p] += 1
  return res

# mean euclidean distance between pixels rgb
def dist_tiles(t_1, t_2):
  dist = 0
  assert(len(t_1) == len(t_2))
  for i in xrange(0,len(t_1)):
    dist += distance.euclidean(t_1[i],t_2[i])
  return dist / len(t_1)

def is_tile_not_relevant(t,T,d):
  for t_ in T:
    dist = dist_tiles(t, t_)
    if dist < d:
      return True
  return False

def max_dist_tiles(a, t):
  max_dist = 0
  assert(len(a) == len(t))
  for i in xrange(0,len(a)):
    dist = distance.euclidean(a[i],t[i])
    if dist > max_dist:
      max_dist = dist
  return max_dist

def get_feature(t, A):
  min_dist = 999999999
  print len(A)
  for a in A:
    dist = max_dist_tiles(a, t)
    if dist < min_dist:
      min_dist = dist
  return min_dist

# adding the darkness of a 5x5 frame
def armar_vector_dark_pix_5x5_color_sin_overlapping(filename):
  im = Image.open(filename)
  pixels = im.load()
  # texture tiles
  T_0 = []
  dim = 5
  for i in range(0,im.size[0]-4,dim):
    for j in range(0,im.size[1]-4,dim):
        frame = [pixels[i,j], pixels[i+1,j], pixels[i+2,j], pixels[i+3,j], pixels[i+4,j],  
        pixels[i,j+1], pixels[i+1,j+1], pixels[i+2,j+1], pixels[i+3,j+1], pixels[i+4,j+1],
        pixels[i,j+2], pixels[i+1,j+2], pixels[i+2,j+2], pixels[i+3,j+2], pixels[i+4,j+2],
        pixels[i,j+3], pixels[i+1,j+3], pixels[i+2,j+3], pixels[i+3,j+3], pixels[i+4,j+3],
        pixels[i,j+4], pixels[i+1,j+4], pixels[i+2,j+4], pixels[i+3,j+4], pixels[i+4,j+4]]
        T_0.append(frame)
  # relevant texture tiles
  T = []
  d = 40.0
  for t in T_0:
    if not is_tile_not_relevant(t,T,d):
      T.append(t)
  print len(T_0), len(T)
  
  # build feature vector
  A = []
  for i in range(0,im.size[0]-4,1):
    for j in range(0,im.size[1]-4,1):
        frame = [pixels[i,j], pixels[i+1,j], pixels[i+2,j], pixels[i+3,j], pixels[i+4,j],
        pixels[i,j+1], pixels[i+1,j+1], pixels[i+2,j+1], pixels[i+3,j+1], pixels[i+4,j+1],
        pixels[i,j+2], pixels[i+1,j+2], pixels[i+2,j+2], pixels[i+3,j+2], pixels[i+4,j+2],
        pixels[i,j+3], pixels[i+1,j+3], pixels[i+2,j+3], pixels[i+3,j+3], pixels[i+4,j+3],
        pixels[i,j+4], pixels[i+1,j+4], pixels[i+2,j+4], pixels[i+3,j+4], pixels[i+4,j+4]]
        A.append(frame)
  res = []
  print "antes"
  for t in T:
      res.append(get_feature(t,T_0))
  print "despues"

  return res

##############################################################

# Armo el listado de imagenes disponibles.
IMGDB = 'train'
NUMIMAGES = 3
filenames_cats = ['%s/cat.%s.jpg' % (IMGDB, str(i))  for i in range(NUMIMAGES)]
filenames_dogs = ['%s/dog.%s.jpg' % (IMGDB, str(i))  for i in range(NUMIMAGES)]

# Extraigo el vector de atributos de cada imagen, y lo guardo en
# un diccionario data[filename] --> vector_atributos
data = {}
y = []
# class 0 is cat, class 1 is dog
for filename in filenames_cats:
  #data[filename] = armar_vector_gris(filename)
  #data[filename] = armar_vector_color(filename)
  #data[filename]  = armar_vector_dark_pix_2x2(filename)
  #data[filename]  = armar_vector_dark_pix_5x5_claro_oscuro(filename)
  #data[filename]  = armar_vector_dark_pix_5x5_claro_oscuro_sin_overlapping(filename)
  data[filename]  = armar_vector_dark_pix_5x5_color_sin_overlapping(filename)
  y.append(0)

for filename in filenames_dogs:
  #data[filename] = armar_vector_gris(filename)
  #data[filename] = armar_vector_color(filename)
  #data[filename] = armar_vector_dark_pix_2x2(filename)
  #data[filename]  = armar_vector_dark_pix_5x5_claro_oscuro(filename)
  #data[filename]  = armar_vector_dark_pix_5x5_claro_oscuro_sin_overlapping(filename)
  data[filename]  = armar_vector_dark_pix_5x5_color_sin_overlapping(filename)
  y.append(1)

# Preparo la matriz con los datos de entrenamiento.
X = [data[filename] for filename in filenames_dogs] + [data[filename] for filename in filenames_cats]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)

print len(X[0])

classifier = ensemble.RandomForestClassifier(n_estimators=256)
classifier.fit(X_train,y_train)
indeces = classifier.apply(X_test)
scores = cross_validation.cross_val_score(classifier, X, y)
print scores.mean()


"""
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
"""


