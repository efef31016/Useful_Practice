import matplotlib.pyplot as plt

class SVMvisualize():

  def __init__(self, X, X_test, y_test, w, b):
    self.X = X
    self.X_test = X_test
    self.y_test = y_test
    self.w = w
    self.b = b

  def visualize_dataset(self,):
      plt.scatter(self.X[:, 0], self.X[:, 1], c=y)


  # Visualizing SVM
  def visualize_svm(self,):

      def get_hyperplane_value(x, w, b, offset):
          return (-w[0][0] * x + b + offset) / w[0][1]

      fig = plt.figure()
      ax = fig.add_subplot(1,1,1)
      plt.scatter(self.X_test[:, 0], self.X_test[:, 1], marker="o", c=self.y_test)

      x0_1 = np.amin(self.X_test[:, 0])
      x0_2 = np.amax(self.X_test[:, 0])

      x1_1 = get_hyperplane_value(x0_1, self.w, self.b, 0)
      x1_2 = get_hyperplane_value(x0_2, self.w, self.b, 0)

      x1_1_m = get_hyperplane_value(x0_1, self.w, self.b, -1)
      x1_2_m = get_hyperplane_value(x0_2, self.w, self.b, -1)

      x1_1_p = get_hyperplane_value(x0_1, self.w, self.b, 1)
      x1_2_p = get_hyperplane_value(x0_2, self.w, self.b, 1)

      ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
      ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
      ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

      x1_min = np.amin(self.X[:, 1])
      x1_max = np.amax(self.X[:, 1])
      ax.set_ylim([x1_min - 3, x1_max + 3])

      plt.show()