import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import shap
import matplotlib.pyplot as plt



#Informações sobre o dataset: https://www.kaggle.com/datasets/uciml/mushroom-classification?resource=download

mushrooms = pd.read_csv("drive/MyDrive/IA/mushrooms.csv")


"""
#Checando se tem dados nulos e vendo informações gerais
mushrooms.describe()
mushrooms['class'].value_counts()
mushrooms.info()

"""

mushrooms['class'] = mushrooms['class'].replace({'e' : 0,'p' : 1}) #Trocando o nome das classes, 0 = "Comestivel" e 1 = "venenoso"

x = mushrooms.iloc[:, 1:22] #separando variáveis independentes
y = mushrooms.iloc[:, 0] #separando variável dependente (classe, queremos classificar)


x = pd.get_dummies(x, drop_first=True) #transforma as colunas que x recebe em puramente numéricas

# =========================================================
# MATRIZ DE CORRELAÇÃO
# =========================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(18,10))

sns.heatmap(
    x.corr(),
    cmap='coolwarm'
)

plt.title("Matriz de Correlação")

plt.show()



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33)


#Normalização dos dados
scaler = MinMaxScaler()

x_train_normalized = scaler.fit_transform(x_train)

x_test_normalized = scaler.transform(x_test)



#metricas
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Algoritmo decisionTree
for i in range(10):

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state= i)

  x_train_normalized = scaler.fit_transform(x_train)

  x_test_normalized = scaler.transform(x_test)

  clf = DecisionTreeClassifier(criterion='gini')
  clf.fit(x_train_normalized, y_train)
  predictx = clf.predict(x_test_normalized)

  acc = accuracy_score(y_test, predictx)
  prec = precision_score(y_test, predictx, average='macro', zero_division=0)
  rec = recall_score(y_test, predictx, average='macro', zero_division=0)
  f1 = f1_score(y_test, predictx, average='macro', zero_division=0)
  accuracies.append(acc)
  precisions.append(prec)
  recalls.append(rec)
  f1_scores.append(f1)

print("\n" + "="*30)
print("MÉDIAS APÓS 10 ITERAÇÕES (DecisionTree)")
print("="*30)
print(f"Média Accuracy:  {np.mean(accuracies):.4f}; Desvio padrão: {np.std(accuracies):.4f}")
print(f"Média Precision: {np.mean(precisions):.4f}; Desvio padrão: {np.std(precisions):.4f}")
print(f"Média Recall:    {np.mean(recalls):.4f}; Desvio padrão: {np.std(recalls):.4f}")
print(f"Média F1 Score:  {np.mean(f1_scores):.4f}; Desvio padrão: {np.std(f1_scores):.4f}")

# Guardando os valores médios do decisionTree.
decision_accuracy = np.mean(accuracies)
decision_precision = np.mean(precisions)
decision_recall = np.mean(recalls)
decision_f1 = np.mean(f1_scores)

accuracies.clear()
precisions.clear()
recalls.clear()
f1_scores.clear()

#algoritmo knn
for i in range(10):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state= i)

    x_train_normalized = scaler.fit_transform(x_train)

    x_test_normalized = scaler.transform(x_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train_normalized, y_train)


    predictx = knn.predict(x_test_normalized)


    acc = accuracy_score(y_test, predictx)
    prec = precision_score(y_test, predictx, average='binary', zero_division=0)
    rec = recall_score(y_test, predictx, average='binary', zero_division=0)
    f1 = f1_score(y_test, predictx, average='binary', zero_division=0)


    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)




print("\n" + "="*30)
print("MÉDIAS APÓS 10 ITERAÇÕES (KNN)")
print("="*30)
print(f"Média Accuracy:  {np.mean(accuracies):.4f}; Desvio padrão: {np.std(accuracies):.4f}")
print(f"Média Precision: {np.mean(precisions):.4f}; Desvio padrão: {np.std(precisions):.4f}")
print(f"Média Recall:    {np.mean(recalls):.4f}; Desvio padrão: {np.std(recalls):.4f}")
print(f"Média F1 Score:  {np.mean(f1_scores):.4f}; Desvio padrão: {np.std(f1_scores):.4f}")

# Guardando as médias do algoritmo KNN.
knn_accuracy = np.mean(accuracies)
knn_precision = np.mean(precisions)
knn_recall = np.mean(recalls)
knn_f1 = np.mean(f1_scores)


accuracies.clear()
precisions.clear()
recalls.clear()
f1_scores.clear()



# Algoritmo SVM
for i in range(10):


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state= i)

    x_train_normalized = scaler.fit_transform(x_train)

    x_test_normalized = scaler.transform(x_test)




    svm_model = SVC(kernel='linear', C=1.0)
    svm_model.fit(x_train_normalized, y_train)


    predict_svm = svm_model.predict(x_test_normalized)

    accuracies.append(accuracy_score(y_test, predict_svm))
    precisions.append(precision_score(y_test, predict_svm, average='macro'))
    recalls.append(recall_score(y_test, predict_svm, average='macro'))
    f1_scores.append(f1_score(y_test, predict_svm, average='macro'))

print("\n" + "="*30)
print("MÉDIAS APÓS 10 ITERAÇÕES (SVM)")
print("="*30)
print(f"Média Accuracy:  {np.mean(accuracies):.4f}; Desvio padrão: {np.std(accuracies):.4f}")
print(f"Média Precision: {np.mean(precisions):.4f}; Desvio padrão: {np.std(precisions):.4f}")
print(f"Média Recall:    {np.mean(recalls):.4f}; Desvio padrão: {np.std(recalls):.4f}")
print(f"Média F1 Score:  {np.mean(f1_scores):.4f}; Desvio padrão: {np.std(f1_scores):.4f}")

# Salvando Médias SVM.
svm_accuracy = np.mean(accuracies)
svm_precision = np.mean(precisions)
svm_recall = np.mean(recalls)
svm_f1 = np.mean(f1_scores)


knn = KNeighborsClassifier(n_neighbors= 5)

knn.fit(x_train_normalized, y_train)

predictx = knn.predict(x_test_normalized)

conf_matrix = confusion_matrix(y_test, predictx)


print('\n\n\n\nMatriz de confusão:')
print(conf_matrix)

"""Fazer matriz de confusãp de todos os algoritmos"""

# =========================================================
# MATRIZ DE CONFUSÃO COM PORCENTAGEM
# =========================================================

conf_matrix_percent = (
    conf_matrix.astype('float')
    / conf_matrix.sum(axis=1)[:, np.newaxis]
) * 100


plt.figure(figsize=(6,5))

sns.heatmap(
    conf_matrix_percent,
    annot=True,
    fmt='.2f',
    cmap='Blues'
)

plt.title("Matriz de Confusão (%)")

plt.xlabel("Previsto")

plt.ylabel("Real")

plt.show()

# =========================================================
# TABELA DE RESULTADOS
# =========================================================

resultados = pd.DataFrame({
    'Modelo': ['DecisionTree', 'KNN', 'SVM'],
    'Accuracy': [decision_accuracy, knn_accuracy, svm_accuracy],
    'Precision': [decision_precision, knn_precision, svm_precision],
    'Recall': [decision_recall, knn_recall, svm_recall],
    'F1-Score': [decision_f1, knn_f1, svm_f1]
})

print("\n\nTabela de Resultados:")
print(resultados)

# =========================================================
# GRÁFICO DE BARRAS
# =========================================================

resultados.plot(
    x='Modelo',
    y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    kind='bar',
    figsize=(10,6)
)

plt.title("Comparação entre os Modelos")

plt.ylabel("Pontuação")

plt.xticks(rotation=0)

plt.show()


# =========================================================
# IMPLEMENTAÇÃO DO SHAP (Explicabilidade do Modelo)
# =========================================================
#foi utilizada a decision tree como base, pois era mais fácil
x_test_df = pd.DataFrame(x_test_normalized, columns=x.columns)


explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(x_test_df)


if isinstance(shap_values, list):
    shap_values_poisonous = shap_values[1]
else:
    shap_values_poisonous = shap_values


plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_poisonous, x_test_df, show=False)
plt.title("Importância das Features para Classe 'Venenoso'", fontsize=14)
plt.tight_layout()
plt.show()



explainer_v2 = shap.Explainer(clf, x_train_normalized, feature_names=x.columns)
shap_values_v2 = explainer_v2(x_test_df)


plt.figure(figsize=(11, 6))

shap.plots.waterfall(shap_values_v2[0, :, 1], show=False)
plt.title("SHAP Waterfall Plot - Explicação do Primeiro Cogumelo do Teste", fontsize=14)
plt.tight_layout()
plt.show()