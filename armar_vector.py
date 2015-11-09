from PIL import Image
from sklearn import cluster, ensemble, cross_validation
import numpy as np
import cv2 
from scipy.spatial import distance
import mahotas as mh

class ArmarVector():

  def __init__(self,filename):
    self.filename = filename

  def armar_vector_color(self):
    COLORS=256    # cantidad de colores para cuantizar las imagenes.
    im = Image.open(self.filename)
    im = im.convert("P", palette=Image.ADAPTIVE, colors=COLORS)
    cantpixels = float(im.size[0] * im.size[1])
    res = [0.0 for i in range(COLORS)]
    for (count, color) in im.getcolors():
        res[color] = count/cantpixels
    return res 

  def armar_vector_gris(self):
    img = cv2.imread(self.filename,0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ)) #stacking images side-by-side
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    histequ,binsequ = np.histogram(equ.flatten(),256,[0,256])
    return histequ

  def __oscuro(self, color_pix):
    #return 1 if color_pix > 128 else 0
    return 1 if color_pix > 200 else 0

    # adding the darkness of a 2x2 frame
  def armar_vector_dark_pix_2x2(self, ):
    res = [0 for i in xrange(0,5)]
    img = cv2.imread(self.filename,0)
    equ = cv2.equalizeHist(img)
    for i in range(equ.shape[0]-1):
      for j in range(equ.shape[1]-1):
        p = self.__oscuro(equ[i,j]) + self.__oscuro(equ[i+1,j]) + self.__oscuro(equ[i+1,j]) + self.__oscuro(equ[i+1,j+1])
        res[p] += 1
    return res

# adding the darkness of a 5x5 frame
  def armar_vector_dark_pix_5x5_claro_oscuro(self, ):
    res = [0 for i in xrange(0,26)]
    img = cv2.imread(self.filename,0)
    equ = cv2.equalizeHist(img)
    for i in range(equ.shape[0]-4):
      for j in range(equ.shape[1]-4):
        p = 0
        for k in xrange(0,5):
          p += self.__oscuro(equ[i,j+k]) + self.__oscuro(equ[i+1,j+k]) + self.__oscuro(equ[i+2,j+k]) + self.__oscuro(equ[i+3,j+k]) + self.__oscuro(equ[i+4,j+k])
        res[p] += 1
    return res

# adding the darkness of a 5x5 frame
  def armar_vector_dark_pix_5x5_claro_oscuro_sin_overlapping(self, ):
    res = [0 for i in xrange(0,26)]
    img = cv2.imread(self.filename,0)
    equ = cv2.equalizeHist(img)
    dim = 5
    for i in range(0,equ.shape[0]-4,dim):
      for j in range(0,equ.shape[1]-4,dim):
        p = 0
        for k in xrange(0,5):
          p += self.__oscuro(equ[i,j+k]) + self.__oscuro(equ[i+1,j+k]) + self.__oscuro(equ[i+2,j+k]) + self.__oscuro(equ[i+3,j+k]) + self.__oscuro(equ[i+4,j+k])
        res[p] += 1
    return res

# mean euclidean distance between pixels rgb
  def __dist_tiles(self, t_1, t_2):
    dist = 0
    assert(len(t_1) == len(t_2))
    for i in xrange(0,len(t_1)):
      dist += distance.euclidean(t_1[i],t_2[i])
    return dist / len(t_1)

  def __is_tile_not_relevant(self, t,T,d):
    for t_ in T:
      dist = self.__dist_tiles(t, t_)
      if dist < d:
        return True
    return False

  def __max_dist_tiles(self, a, t):
    max_dist = 0
    assert(len(a) == len(t))
    for i in xrange(0,len(a)):
      dist = distance.euclidean(a[i],t[i])
      if dist > max_dist:
        max_dist = dist
    return max_dist

  def __get_feature(self,t, A):
    min_dist = 999999999
    print len(A)
    for a in A:
      dist = self.__max_dist_tiles(a, t)
      if dist < min_dist:
        min_dist = dist
    return min_dist

# adding the darkness of a 5x5 frame
  def armar_vector_dark_pix_5x5_color_sin_overlapping(self, ):
    im = Image.open(self.filename)
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
      if not self.__is_tile_not_relevant(t,T,d):
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
      res.append(self.__get_feature(t,T_0))
    print "despues"

    return res

  def armar_vector_mahotas(self, ):
    img = mh.imread(self.filename,as_grey=True)
    return mh.features.lbp(img, 8, 12, ignore_zeros=False) # 8 12

  def armar_vector_mix(self, ):
    return self.armar_vector_dark_pix_5x5_claro_oscuro_sin_overlapping() + self.armar_vector_mahotas().tolist()
  # armar_vector_dark_pix_2x2 no suma
  # armar_vector_color empeora mucho



