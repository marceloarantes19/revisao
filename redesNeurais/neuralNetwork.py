import numpy as np
class NeuralNetwork:
  def __init__(self, camadas, alpha=0.1):
    self.W = []
    self.camadas = camadas
    self.alpha = alpha
    for i in np.arange(0, len(camadas) - 2):
      w = np.random.randn(camadas[i] + 1, camadas [i + 1] + 1)
      self.W.append(w / np.sqrt(camadas[i]))
    w = np.random.randn(camadas[-2] + 1, camadas[-1])
    self.W.append(w / np.sqrt(camadas[-2]))
  
  def __repr__(self):
    return "Rede Neural: {}".format("-".join(str(l) for l in self.camadas))
  
  def sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))
  
  def sigmoidDerivativo(self, x):
    return x * (1.0 - x)

  def fit(self, X, y, epochs = 100, displayUpdate = 100):
    X = np.c_[X, np.ones((X.shape[0]))]
    for epoch in range(0, epochs):
      for (x, target) in zip(X, y):
        self.fitParcial(x, target)
      
      if epoch == 0 or ((epoch + 1) % displayUpdate)== 0:
        perda = self.calculaPerda(X, y)
        print("[Info] epoca={}, perda={:.7f}".format(epoch + 1, perda))

  def fitParcial(self, x, y):
    A = [np.atleast_2d(x)]
    for camada in np.arange(0, len(self.W)):
      net = A[camada].dot(self.W[camada])
      out = self.sigmoid(net)
      A.append(out)
    erro = A[-1] - y
    D = [erro * self.sigmoidDerivativo(A[-1])]
    for camada in np.arange(len(A) -2, 0, -1):
      delta = D[-1].dot(self.W[camada].T)
      delta = delta * self.sigmoidDerivativo(A[camada])
      D.append(delta)
    D = D[::-1]
    for camada in np.arange(0, len(self.W)):
      self.W[camada] += -self.alpha*A[camada].T.dot(D[camada])
  
  def predict(self, X, addBias=True):
    p = np.atleast_2d(X)
    if addBias:
      p = np.c_[p, np.ones((p.shape[0]))]
    for camada in np.arange(0, len(self.W)):
      p = self.sigmoid(np.dot(p, self.W[camada]))
    return p
  
  def calculaPerda(self, X, targets):
    targets = np.atleast_2d(targets)
    predictions = self.predict(X, addBias=False)
    perda = 0.5 *np.sum((predictions-targets)**2) 
    return perda
      