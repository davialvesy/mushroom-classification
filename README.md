# 🍄 Classificação de Cogumelos com Machine Learning

Trabalho avaliativo em grupo desenvolvido na disciplina de Inteligência Artificial da graduação, com o objetivo de classificar cogumelos como comestíveis ou venenosos utilizando técnicas de Machine Learning.

## 📖 Sobre o projeto

O projeto utiliza o dataset [Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification) do Kaggle para treinar e comparar diferentes algoritmos de classificação, avaliando qual apresenta o melhor desempenho na tarefa de identificar cogumelos venenosos a partir de suas características físicas. Além da classificação, o trabalho também explora a explicabilidade dos modelos, buscando entender quais atributos mais influenciam as previsões.

O desenvolvimento foi realizado na ferramenta **Google Colab**.

## 🚀 Funcionalidades

* Pré-processamento e transformação dos dados (codificação de variáveis categóricas, normalização);
* Análise exploratória com matriz de correlação;
* Treinamento e comparação de três algoritmos de classificação:
  * Decision Tree
  * K-Nearest Neighbors (KNN)
  * Support Vector Machine (SVM)
* Validação dos modelos em múltiplas iterações, com cálculo de média e desvio padrão das métricas;
* Avaliação de desempenho através de Acurácia, Precisão, Recall e F1-Score;
* Visualização dos resultados com matriz de confusão e gráfico comparativo entre os modelos;
* Explicabilidade do modelo com **SHAP** (SHapley Additive exPlanations), incluindo gráfico de importância de features e waterfall plot.

## 🛠️ Tecnologias utilizadas

* Python
* Google Colab
* Pandas / NumPy
* Scikit-learn
* Seaborn / Matplotlib
* SHAP

## 📊 Metodologia

Para cada algoritmo, o dataset foi dividido em treino e teste (67%/33%) e o processo foi repetido em 10 iterações com diferentes sementes aleatórias (`random_state`), garantindo uma avaliação mais robusta e menos sensível a uma única divisão dos dados. As métricas de cada rodada foram armazenadas e, ao final, calculada a média e o desvio padrão de cada uma.

## 📈 Explicabilidade (SHAP)

Após a comparação dos modelos, foi utilizada a árvore de decisão como base para uma análise de explicabilidade com SHAP, permitindo identificar quais características dos cogumelos mais contribuíram para a classificação como "venenoso", tanto de forma geral (summary plot) quanto para um caso individual (waterfall plot).

## ▶️ Como executar

1. Faça o download do dataset [Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification) do Kaggle;
2. Faça upload do arquivo `mushrooms.csv` para o seu Google Drive;
3. Abra o notebook no Google Colab;
4. Ajuste o caminho do arquivo na linha de leitura do dataset, se necessário:
   ```python
   mushrooms = pd.read_csv("drive/MyDrive/IA/mushrooms.csv")
   ```
5. Execute as células do notebook em ordem.

## 📚 Conceitos aplicados

* Pré-processamento de dados categóricos (One-Hot Encoding)
* Normalização de dados (MinMaxScaler)
* Classificação supervisionada
* Validação com múltiplas iterações
* Métricas de avaliação de modelos (Acurácia, Precisão, Recall, F1-Score)
* Explicabilidade de modelos (XAI) com SHAP
