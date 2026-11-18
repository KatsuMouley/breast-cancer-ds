import pandas as pd
from pickle import load

#nova paciente
nova_paciente = {
    'age': '50-59',
    'menopause': 'ge40',
    'tumor-size': '30-34',
    'inv-nodes': '3-5',
    'node-caps': 'yes',
    'deg-malig': 3,
    'breast': 'left',
    'breast-quad': 'left_low',
    'irradiat': 'yes'
}
nova_paciente = pd.DataFrame([nova_paciente])

#carregar o normalizador, as colunas e o modelo salvos no treinamento
normalizador = load(open('normalizador_breast.model', 'rb'))
colunas = load(open('colunas_breast.model', 'rb'))
breast_tree = load(open('breast_tree.model', 'rb'))

#normalizar a nova paciente do mesmo jeito do treino
paciente_num = pd.DataFrame(normalizador.transform(nova_paciente[['deg-malig']]), columns=['deg-malig'])
paciente_cat = pd.get_dummies(nova_paciente.drop(columns=['deg-malig']), dtype=int)
paciente_normalizada = paciente_num.join(paciente_cat)
#deixar com as mesmas colunas do treino (as que nao aparecem ficam 0)
paciente_normalizada = paciente_normalizada.reindex(columns=colunas, fill_value=0)

#classificar a nova paciente
diagnostico = breast_tree.predict(paciente_normalizada)
print('Diagnostico (Class):', diagnostico[0])
