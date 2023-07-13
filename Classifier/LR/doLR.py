import numpy as np

class LogisticRegression():
    def __init__(self, x, w, b, target):
        self.x = x
        self.w = w
        self.b = b
        self.target = target

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def forward(self, data):
        y_pred = self.sigmoid(np.dot(data, self.w) + self.b).reshape(-1, 1)
        return y_pred

    def loss(self, y_pred, target):
      x = target*np.log(y_pred) + (1-target)*np.log(1-y_pred)
      return -(np.mean(x))

    def optimizer(self, y_pred, lr, data, target):
        self.w = self.w - lr * np.mean(data * (y_pred - target), axis=0)
        self.b = self.b - lr * np.mean((y_pred - target), axis=0)

        return self.w, self.b

    def fit(self, num_epochs, lr):
        loss_save=[]
        for epoch in range(num_epochs):
            for i, data in enumerate(self.x):
              # forward pass and loss
              y_pred = self.forward(data)
              loss = self.loss(y_pred, self.target[i])
              loss_save.append(loss)
              # update
              self.optimizer(y_pred, lr, self.x[i], self.target[i])

            if (epoch+1) % 10 == 0:
              print(f'epoch {epoch + 1}: loss = {loss}')

        return loss_save