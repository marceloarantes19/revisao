import pandas as pd
import numpy as np
from neuralNetworkTanh import NeuralNetwork 

# Inserindo os dados do seno no formato csv
X = []
y = []
df = pd.read_csv("E:\\aulas\\TE2\\revisao\\redesNeurais\\seno.csv", encoding = "UTF-8", sep=";")
print(df.shape)
for i in df.index:
  j = []
  k = [] 
  j.append(df['XRad'][i])
  k.append(df['SenoX'][i])
  X.append(j)
  y.append(k)

X = np.array(X)
y = np.array(y)

print("Treinando o perceptron")
nn = NeuralNetwork([1, 7, 1], alpha=0.12)
nn.fit(X, y, epochs=15000)

print("Testando o perceptron")
for (x, target) in zip(X, y):
  pred = nn.predict(x)[0][0]
  print("[INFO] data = {}, ground-truth = {}, pred = {:.4f}".format(x, target[0], pred))