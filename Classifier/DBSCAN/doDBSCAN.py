class DBCSAN:
  def __init__(self, X, eps, minpts):
    self.X = np.concatenate((X, np.zeros((X.shape[0],1))), axis=1)
    self.eps = eps
    self.minpts = minpts

    print("X.shape:",X.shape)
    print("X[-2]: status (0: no-visited; 1: noise; 2: belongs to some class)")
    print("We determine the magnitude of this number based on the order of discovery of its category, where n >= 2 represents the (n-1)-th discovered category.")

  def dist(self, x, y):
    return np.sqrt(sum((x-y)**2))

  def regionQuery(self, P):

    '''
    P: index of point P
    '''

    neighbors = []
    for Pprime in range(len(self.X)):
      if self.dist(self.X[P][:-1], self.X[Pprime][:-1]) < self.eps:
        neighbors.append(Pprime)

    return neighbors

  def fit(self,):

    self.update_save = []
    C = 2

    for i, j in enumerate(self.X):

      if j[-1] != 0:
        continue

      nb = self.regionQuery(i)

      if len(nb) < self.minpts:
        j[-1] = 1  # noise point

      else:
        j[-1] = C
        k = 0

        while k < len(nb):

          if self.X[nb[k]][-1] == 1:
            self.X[nb[k]][-1] = C

          elif self.X[nb[k]][-1] == 0:
            self.X[nb[k]][-1] = C
            Pprime = self.regionQuery(nb[k])
            if len(Pprime) >= self.eps:
              nb += Pprime

          self.update_save.append(self.X[:,-1].copy())
          k+=1
      C+=1

    return self.X

def Dbscan_process(save):
  for i in range(len(save)):
    if i%100 == 0:
      plt.scatter(X[:,0], X[:,1], c=save[i])
      plt.savefig("dbscan_%d.png"%i)
    elif i == len(save)-1:
      plt.scatter(X[:,0], X[:,1], c=save[i])
      plt.savefig("dbscan_%d.png"%((i//100+1)*100))
    else:
      pass