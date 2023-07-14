import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio

class Kmeans():

  def __init__(self, k, X):
    self.k = k
    self.X = X

  def twonorm(self, x,y):
    return np.sqrt(np.sum((x-y)**2))

  def fit(self, Tol):

    self.initcenter = np.random.uniform(np.min(self.X), np.max(self.X), size=(self.k ,self.X.shape[1]))
    self.center_save = [self.initcenter.copy()]
    center = self.initcenter.copy()
    while True:
      clscorX = np.zeros(self.X.shape[0]) # 當前每個點對應到的 center
      tmpcenter = center.copy()
      for i in range(len(self.X)):
        dist = np.inf
        for j in range(self.k):
          if self.twonorm(self.X[i], center[j]) <= dist:
            dist = self.twonorm(self.X[i], center[j])
            clscorX[i] = j
      for j in range(self.k):
        center[j] = np.mean(self.X[np.where(clscorX==j)], axis=0)
      self.center_save.append(center.copy())

      if self.twonorm(tmpcenter, center) < Tol:
        print("best centers are\n", center)
        break


# Note : the two functions below only for this case
def process(center):
  for i in range(len(center)):
    plt.scatter(X[:, 0], X[:, 1], c = y)
    plt.scatter([center[i][0,0], center[i][1,0]], [center[i][0,1], center[i][1,1]], c='red', s=70)
    plt.title("4 step update")
    plt.savefig("%d.png"%i, format='png')
    plt.show()

def gif(image_files):
  images = [Image.open(file) for file in image_files]
  output_file = 'update.gif'
  imageio.mimsave(output_file, images, duration=0.5)