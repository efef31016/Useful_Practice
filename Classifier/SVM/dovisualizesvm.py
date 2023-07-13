import matplotlib.pyplot as plt

class SVMvisualize():

  def __init__(self, X_all, y_all , X_test, y_test, w, b, feature_i, feature_j):
    self.X = X_all
    self.y = y_all
    self.X_test = X_test
    self.y_test = y_test
    self.w = w
    self.b = b
    self.i = feature_i
    self.j = feature_j

  def visualize_dataset(self,):
      plt.scatter(self.X[:, self.i], self.X[:, self.j], c = self.y)
      plt.title("dataset")

  # Visualizing SVM
  def visualize_svm(self,):
    
      # ax + by + c = 0 → y = (-ax - c) / b : 示意圖
      def get_hyperplane_value(x, w, b, offset):
          return (-w[0][self.i] * x - b + offset) / w[0][self.j]

      fig = plt.figure()
      ax = fig.add_subplot(1,1,1)
      plt.scatter(self.X_test[:, self.i], self.X_test[:, self.j], marker="o", c=self.y_test)

      x0_1 = np.amin(self.X_test[:, self.i])
      x0_2 = np.amax(self.X_test[:, self.i])

      x1_1 = get_hyperplane_value(x0_1, self.w, self.b, 0)
      x1_2 = get_hyperplane_value(x0_2, self.w, self.b, 0)

      x1_1_m = get_hyperplane_value(x0_1, self.w, self.b, -0.2)
      x1_2_m = get_hyperplane_value(x0_2, self.w, self.b, -0.2)

      x1_1_p = get_hyperplane_value(x0_1, self.w, self.b, 0.2)
      x1_2_p = get_hyperplane_value(x0_2, self.w, self.b, 0.2)

      ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
      ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
      ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

      x1_min = np.amin(self.X[:, self.j])
      x1_max = np.amax(self.X[:, self.j])
      ax.set_ylim([x1_min - 3, x1_max + 3])
      ax.set_title("test data")

      plt.show()
