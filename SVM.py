import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

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

# Criar e treinar o modelo SVM
svm_model = SVC(kernel="rbf", C=1.0, gamma="auto", random_state=42)
svm_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = svm_model.predict(X_test)

# Avaliar o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Comestível", "Venenoso"], yticklabels=["Comestível", "Venenoso"])
plt.xlabel("Previsão")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.show()

# Relatório de classificação
print("Relatório de Classificação:\n", classification_report(y_test, y_pred, target_names=["Comestível", "Venenoso"]))

# Validação cruzada
cv_scores = cross_val_score(svm_model, X_encoded, y_encoded, cv=5, scoring="accuracy")
print("Acurácia média na validação cruzada:", cv_scores.mean())
print("Desvio padrão na validação cruzada:", cv_scores.std())

print("Número total de instâncias:", len(data))
print("Número de instâncias após pré-processamento:", len(X_encoded))
