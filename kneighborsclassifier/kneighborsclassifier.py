import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# dividindo treinamento e teste
train = pd.read_csv('../train.csv')
target = train['class'].values
features = train.drop(['class', 'suit_1', 'suit_2',
                       'suit_3', 'suit_4', 'suit_5'], axis=1).values
test = pd.read_csv('../test.csv')
test = test.dropna()
tfeatures = test.drop(['class', 'suit_1', 'suit_2',
                       'suit_3', 'suit_4', 'suit_5'], axis=1).values
ttarget = test['class'].values

# realizando treinamento e avaliando predição
pred = KNeighborsClassifier(n_neighbors=20, weights='distance')
pred.fit(features, target)
print(pred.score(tfeatures, ttarget))

# pequeno exemplo de predição
print(tfeatures[[5]])
print(pred.predict(tfeatures[[5]]))
