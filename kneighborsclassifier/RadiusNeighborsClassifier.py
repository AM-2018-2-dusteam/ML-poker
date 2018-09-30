import pandas as pd
from sklearn.neighbors import RadiusNeighborsClassifier

# dividindo treinamento e teste
train = pd.read_csv('../train.csv')
target = train['class'].values
features = train.drop('class', axis=1).values
test = pd.read_csv('../test.csv')
tfeatures = test.drop('class', axis=1).values
ttarget = test['class'].values

# realizando treinamento e avaliando predição
pred = RadiusNeighborsClassifier(radius=3.0, weights='distance')
pred.fit(features, target)
print('confiabilidade da predição em uma escala de 0 a 1:',
      pred.score(tfeatures, ttarget))

# pequeno exemplo de predição
print('cartas:', tfeatures[[5]])
print('mao de poker:', pred.predict(tfeatures[[5]]))
