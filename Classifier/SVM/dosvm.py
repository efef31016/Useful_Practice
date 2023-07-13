# svm.py
import numpy as np


class SVM:

    def __init__(self, alpha = 1.0):
        self.alpha = alpha
        self.w = 0
        self.b = 0

    # Hinge Loss Function / Calculation
    def hingeloss(self, w, b, x, y):

        loss = 0.5 * np.sum(w**2)
        for i in range(x.shape[0]):
          opt_term = y[i] * ((np.dot(w, x[i])) + b)
          loss += max(0, 1-opt_term)

        return self.alpha * loss

    def fit(self, X, Y, batch_size=100, learning_rate=0.001, epochs=1000):
        # The number of features in X
        number_of_features = X.shape[1]

        # The number of Samples in X
        number_of_samples = X.shape[0]

        c = self.alpha

        # # Shuffling the samples randomly
        ids = np.arange(number_of_samples)
        np.random.shuffle(ids)

        # parameters
        w = np.zeros((1, number_of_features))
        b = 0
        loss_save = []

        # Stochastic Gradient Descent
        for i in range(epochs):

            l = self.hingeloss(w, b, X, Y)
            loss_save.append(l)

            # (number_of_samples // batch_size) data
            for batch_initial in range(0, number_of_samples, batch_size):

                gradw = 0
                gradb = 0

                for j in range(batch_initial, batch_initial + batch_size):
                    if j < number_of_samples:

                        x = ids[j]
                        ti = Y[x] * (np.dot(w, X[x]) + b)

                        if ti <= 1:
                          # w
                          gradw += c * Y[x] * X[x]
                          # b
                          gradb += c * Y[x]

                # Updating weights and bias
                w = w - learning_rate * (w - gradw)
                b = b + learning_rate * gradb

        self.w = w
        self.b = b

        return self.w, self.b, loss_save

    def predict(self, X):
        prediction = np.dot(X, self.w[0]) + self.b
        return np.sign(prediction)