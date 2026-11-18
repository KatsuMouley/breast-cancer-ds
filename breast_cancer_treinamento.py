import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV #hiperparametrizador
from sklearn.metrics import accuracy_score #acuracia do modelo
from sklearn.tree import DecisionTreeClassifier #Algoritmo classificador
from imblearn.over_sampling import SMOTE #Para balancear os dados
from pickle import dump

#1 CARREGAR OS DADOS
colunas = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps',
           'deg-malig', 'breast', 'breast-quad', 'irradiat']
dados = pd.read_csv('breast-cancer.data', names=colunas, na_values='?')
print(dados['Class'].value_counts())

#2 PREENCHER OS DADOS FALTANTES (node-caps e breast-quad tem '?')
#como sao categoricos, preenche com a moda
dados = dados.fillna(dados.mode().iloc[0])

#3 SEPARAR ATRIBUTOS E CLASSE
dados_atributos = dados.drop(columns=['Class'])
dados_classes = dados['Class']

#4 NORMALIZAR OS DADOS
#separar numericos dos categoricos
dados_num = dados_atributos[['deg-malig']]
dados_cat = dados_atributos.drop(columns=['deg-malig'])

#numericos: MinMax
normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_num)
dump(modelo_normalizador, open('normalizador_breast.model', 'wb'))
dados_num_normalizados = pd.DataFrame(modelo_normalizador.transform(dados_num), columns=dados_num.columns)

#categoricos: get_dummies
dados_cat_normalizados = pd.get_dummies(dados_cat, dtype=int)

#juntar tudo
dados_atributos_normalizados = dados_num_normalizados.join(dados_cat_normalizados)
#salvar as colunas para usar na inferencia
dump(list(dados_atributos_normalizados.columns), open('colunas_breast.model', 'wb'))

#5 SEPARAR TREINO E TESTE
atributos_treino, atributos_teste, classes_treino, classes_teste = train_test_split(
    dados_atributos_normalizados, dados_classes, test_size=0.3, random_state=42, stratify=dados_classes)

#6 BALANCEAR OS DADOS (somente o treino)
resampler = SMOTE(random_state=42)
atributos_treino_balanceados, classes_treino_balanceadas = resampler.fit_resample(atributos_treino, classes_treino)

#7 HIPERPARAMETRIZACAO
criterion = ['gini', 'entropy', 'log_loss']
splitter = ['best', 'random']
max_depth = [None, 3, 4, 5, 6, 8]
min_samples_split = [int(x) for x in np.linspace(start=2, stop=20, num=10)]

decision_tree_grid = {
    'criterion': criterion,
    'splitter': splitter,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split
}

#testar com os dados balanceados e sem balancear, e ficar com o de melhor acuracia
testes = {
    'balanceado': (atributos_treino_balanceados, classes_treino_balanceadas),
    'nao balanceado': (atributos_treino, classes_treino)
}

melhor_acuracia = 0
for nome, (atributos, classes) in testes.items():
    tree = DecisionTreeClassifier(random_state=42)
    tree_hiperparametros = RandomizedSearchCV(
        estimator=tree,
        param_distributions=decision_tree_grid,
        n_iter=30,
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    tree_hiperparametros.fit(atributos, classes)

    #reconstruir o classificador com os melhores parametros e treinar
    tree = DecisionTreeClassifier(**tree_hiperparametros.best_params_, random_state=42)
    breast_tree = tree.fit(atributos, classes)

    #avaliar a acuracia com os dados de teste
    classes_preditas = breast_tree.predict(atributos_teste)
    acuracia = accuracy_score(classes_teste, classes_preditas)
    print(nome, '- parametros:', tree_hiperparametros.best_params_)
    print(nome, '- acuracia:', acuracia)

    if acuracia > melhor_acuracia:
        melhor_acuracia = acuracia
        melhor_modelo = breast_tree

#8 SALVAR O MELHOR MODELO
dump(melhor_modelo, open('breast_tree.model', 'wb'))
print('Melhor acuracia:', melhor_acuracia)
