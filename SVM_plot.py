import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Carregar o dataset
caminho_arquivo = "C:\\Users\\Arlison Gaspar\\Desktop\\Projetos Visual studio\\Inteligencia Artificial\\TRabalho p3\\agaricus-lepiota.data"
column_names = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]
data = pd.read_csv(caminho_arquivo, header=None, names=column_names)
data = data.replace("?", "missing")

# Selecionar as características mais discriminantes
features = ["odor", "spore-print-color", "stalk-surface-below-ring", "habitat", "cap-color", "population"]
X = data[features]
y = data["class"]

# Converter atributos categóricos para numéricos
label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)
y_encoded = label_encoder.fit_transform(y)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.7, random_state=42)

# Reduzir a dimensionalidade para 2D usando PCA (análise de componentes principais)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_encoded)

# Criar e treinar o modelo SVM com as 6 características
svm_model = SVC(kernel="rbf", C=1.0, gamma="auto", random_state=42)
svm_model.fit(X_train, y_train)

# Plotar o gráfico de dispersão
plt.figure(figsize=(8, 6))

# Plotar os pontos de cada classe
plt.scatter(X_pca[y_encoded == 0, 0], X_pca[y_encoded == 0, 1], color="blue", label="Comestível", alpha=0.6)
plt.scatter(X_pca[y_encoded == 1, 0], X_pca[y_encoded == 1, 1], color="red", label="Venenoso", alpha=0.6)

# Plotar o hiperplano de decisão
xx, yy = np.meshgrid(np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 500),
                     np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 500))

# Aplicar PCA também na malha de pontos (xx, yy)
Z = svm_model.decision_function(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Plotar as margens
plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["black", "orange", "black"], linestyles=["--", "-", "--"])
plt.title("SVM com Margem e Hiperplano de Decisão")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend()
plt.show()
