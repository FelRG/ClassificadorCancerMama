# Classificador de Câncer de Mama com Árvore de Decisão

## Descrição

Este projeto utiliza Machine Learning para classificar casos de câncer de mama como benignos ou malignos.

O modelo foi construído com Python utilizando o algoritmo de Árvore de Decisão e o dataset `Breast Cancer Wisconsin` disponível na biblioteca Scikit-Learn.

---

## Bibliotecas Utilizadas

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
```

---

## Dataset Utilizado

O projeto utiliza o dataset de câncer de mama disponível no Scikit-Learn:

```python
breast_cancer = load_breast_cancer()
```

O dataset contém:

- Características extraídas de exames
- Classes de diagnóstico
- Casos benignos e malignos

---

## Etapas do Projeto

### 1. Carregamento dos Dados

```python
breast_cancer = load_breast_cancer()
```

---

### 2. Separação das Variáveis

```python
X = breast_cancer.data
y = breast_cancer.target
```

- `X`: características dos exames
- `y`: classe de diagnóstico

---

### 3. Divisão entre Treino e Teste

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

- 70% dos dados para treino
- 30% dos dados para teste

---

### 4. Criação do Modelo

```python
model = DecisionTreeClassifier(random_state=42, max_depth=9)
```

Foi utilizado o algoritmo de Árvore de Decisão com profundidade máxima de 9 níveis.

---

### 5. Treinamento do Modelo

```python
model.fit(X_train, y_train)
```

---

### 6. Realização das Previsões

```python
predictions = model.predict(X_test)
```

O modelo realiza previsões para identificar se o caso é benigno ou maligno.

---

## Métricas Avaliadas

### Acurácia

```python
accuracy_score(y_test, predictions)
```

Mede a porcentagem total de acertos do modelo.

---

### Matriz de Confusão

```python
confusion_matrix(y_test, predictions)
```

Permite identificar:

- Verdadeiros positivos
- Verdadeiros negativos
- Falsos positivos
- Falsos negativos

---

### Precisão, Recall e F1-Score

```python
precision_score(y_test, predictions, average='weighted')
recall_score(y_test, predictions, average='weighted')
f1_score(y_test, predictions, average='weighted')
```

Essas métricas ajudam a avaliar a qualidade do modelo de forma mais detalhada.

---

## Resultados

O modelo apresentou aproximadamente 92% de acurácia nas previsões.

Apesar do bom desempenho, ainda existem alguns falsos negativos, o que é importante neste tipo de problema, pois pode indicar casos em que o modelo prevê que a pessoa está saudável quando na verdade possui a doença.

---

## Possíveis Melhorias

- Ajustar parâmetros da árvore, como `max_depth`
- Testar outros algoritmos de classificação
- Utilizar Random Forest
- Realizar validação cruzada
- Fazer balanceamento de classes
- Comparar diferentes métricas de desempenho

---

## Como Executar

1. Instale as dependências:

```bash
pip install scikit-learn matplotlib seaborn pandas
```

2. Execute o notebook normalmente.

---

## Tecnologias Utilizadas

- Python
- Scikit-Learn
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook
