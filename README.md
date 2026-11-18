# Trabalho Breast Cancer - Data Science

Base de dados: https://archive.ics.uci.edu/dataset/14/breast+cancer

- `breast_cancer_treinamento.py`: modulo de treinamento (preenchimento, normalizacao, SMOTE, hiperparametrizacao da DecisionTree e escolha do modelo de melhor acuracia)
- `breast_cancer_inferencia.py`: modulo de inferencia, recebe uma nova paciente e retorna o diagnostico (Class)

## Como rodar

```
python -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
python breast_cancer_treinamento.py
python breast_cancer_inferencia.py
```
