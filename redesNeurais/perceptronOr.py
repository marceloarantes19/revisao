from Perceptron import Perceptron
import numpy as np

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

print("Treinando o perceptron")
p = Perceptron(X.shape[1], alpha=1.5)
p.fit(X, y, epochs=20000)

print("Testando o perceptron")
for (x, target) in zip(X, y):
  pred = p.predict(x)
  print("[INFO] data = {}, ground-truth = {}, pred = {}".format(x, target[0], pred))