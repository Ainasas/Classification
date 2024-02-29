import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle

base_dados = pd.read_csv('apple_quality.csv')

base_dados = base_dados.drop(index=4000, axis=0)

x = base_dados.iloc[:, 1:7].values
y = base_dados.iloc[:, 8].values

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.25, random_state=1)

rf = RandomForestClassifier(n_estimators= 100, random_state=1, criterion='gini')
rf.fit(x_treino, y_treino)
prev_floresta = rf.predict(x_teste)

svm = SVC(kernel = 'rbf', C=2, random_state=1)
svm.fit(x_treino, y_treino)
prev_svm = svm.predict(x_teste)

rn = MLPClassifier(max_iter = 900, hidden_layer_sizes=200)
rn.fit(x_treino, y_treino)
prev_rede_neural = rn.predict(x_teste)

print(f"""
      A precisão de cada classificador é:
      Random Florest = {accuracy_score(y_teste, prev_floresta)}
      Svm = {accuracy_score(y_teste, prev_svm)}
      Rede Neural = {accuracy_score(y_teste, prev_rede_neural)}""")