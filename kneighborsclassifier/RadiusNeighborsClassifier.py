import pandas as pd
from sklearn.neighbors import RadiusNeighborsClassifier

# dividindo treinamento e teste
train = pd.read_csv('../train.csv')
target = train['class'].values
features = train.drop(['class', 'suit_1', 'suit_2',
                       'suit_3', 'suit_4'], axis=1).values
test = pd.read_csv('../test.csv')
test = test.dropna()
tfeatures = test.drop(['class', 'suit_1', 'suit_2',
                       'suit_3', 'suit_4'], axis=1).values
ttarget = test['class'].values

# realizando treinamento e avaliando predição
pred = RadiusNeighborsClassifier(radius=3.5, weights='distance')
pred.fit(features, target)
print('confiabilidade da predição em uma escala de 0 a 1:',
      pred.score(tfeatures, ttarget))

# pequeno exemplo de predição
print('cartas:', tfeatures[[5]])
print('mao de poker:', pred.predict(tfeatures[[5]]))
