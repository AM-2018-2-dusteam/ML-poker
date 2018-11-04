import inline as inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import f_regression
import warnings
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
#%matplotlib inline

#carregando dados
#df = pd.read_csv("test.csv")
#df = pd.read_csv("train.csv")
df = pd.read_csv("total.csv")
warnings.filterwarnings("ignore")

###Informações do DataFrame
# print(df.shape)
# print()
# print(df.head(5))
# print()
# print(df.tail(5))
# print()

#verificando sé há algum valor nulo
print("Valor nulo: ", df.isnull().values.any())

# Identificando a correlação entre as variáveis
def plot_corr(df, size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize = (size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()
#plota o gráfico da correlação entre as váriaveis
#plot_corr(df)
#tabela das corelações
#print("Correlações das Variáveis")
#print(df.corr())
print()

# Verificando como os dados estão distribuídos
num_nothing_in_hand = len(df.loc[df['class'] == 0])
num_One_pair = len(df.loc[df['class'] == 1])
num_Two_pairs = len(df.loc[df['class'] == 2])
num_Three_of_a_kind = len(df.loc[df['class'] == 3])
num_Straight = len(df.loc[df['class'] == 4])
num_Flush = len(df.loc[df['class'] == 5])
num_Full_house = len(df.loc[df['class'] == 6])
num_Four_of_a_kind = len(df.loc[df['class'] == 7])
num_Straight_flush = len(df.loc[df['class'] == 8])
num_Royal_flush = len(df.loc[df['class'] == 9])

#Imprime esses dados
# print("Numero de  Nothing in Hand: {0} ({1:2.2f}%)".format(num_nothing_in_hand, (num_One_pair/ (num_One_pair+num_Flush+num_Four_of_a_kind+num_Full_house+num_Royal_flush+num_Straight+num_Straight_flush+num_Three_of_a_kind+num_Two_pairs+num_nothing_in_hand))*100))
# print("Numero de  One pair: {0} ({1:2.2f}%)".format(num_One_pair, (num_One_pair/ (num_One_pair+num_Flush+num_Four_of_a_kind+num_Full_house+num_Royal_flush+num_Straight+num_Straight_flush+num_Three_of_a_kind+num_Two_pairs+num_nothing_in_hand))*100))
# print("Numero de  Two pairs: {0} ({1:2.2f}%)".format(num_Two_pairs, (num_Two_pairs/ (num_One_pair+num_Flush+num_Four_of_a_kind+num_Full_house+num_Royal_flush+num_Straight+num_Straight_flush+num_Three_of_a_kind+num_Two_pairs+num_nothing_in_hand))*100))
# print("Numero de  Three of a kind: {0} ({1:2.2f}%)".format(num_Three_of_a_kind, (num_Three_of_a_kind/ (num_One_pair+num_Flush+num_Four_of_a_kind+num_Full_house+num_Royal_flush+num_Straight+num_Straight_flush+num_Three_of_a_kind+num_Two_pairs+num_nothing_in_hand))*100))
# print("Numero de  Straight: {0} ({1:2.2f}%)".format(num_Straight, (num_Straight/ (num_One_pair+num_Flush+num_Four_of_a_kind+num_Full_house+num_Royal_flush+num_Straight+num_Straight_flush+num_Three_of_a_kind+num_Two_pairs+num_nothing_in_hand))*100))
# print("Numero de  Flush: {0} ({1:2.2f}%)".format(num_Flush, (num_Flush/ (num_One_pair+num_Flush+num_Four_of_a_kind+num_Full_house+num_Royal_flush+num_Straight+num_Straight_flush+num_Three_of_a_kind+num_Two_pairs+num_nothing_in_hand))*100))
# print("Numero de  Full House: {0} ({1:2.2f}%)".format(num_Full_house, (num_Three_of_a_kind/ (num_One_pair+num_Flush+num_Four_of_a_kind+num_Full_house+num_Royal_flush+num_Straight+num_Straight_flush+num_Three_of_a_kind+num_Two_pairs+num_nothing_in_hand))*100))
# print("Numero de  Four of a kind: {0} ({1:2.2f}%)".format(num_Four_of_a_kind, (num_Three_of_a_kind/ (num_One_pair+num_Flush+num_Four_of_a_kind+num_Full_house+num_Royal_flush+num_Straight+num_Straight_flush+num_Three_of_a_kind+num_Two_pairs+num_nothing_in_hand))*100))
# print("Numero de  Straight Flush: {0} ({1:2.2f}%)".format(num_Straight_flush, (num_Three_of_a_kind/ (num_One_pair+num_Flush+num_Four_of_a_kind+num_Full_house+num_Royal_flush+num_Straight+num_Straight_flush+num_Three_of_a_kind+num_Two_pairs+num_nothing_in_hand))*100))
# print("Numero de  Royal Flush: {0} ({1:2.2f}%)".format(num_Royal_flush, (num_Three_of_a_kind/ (num_One_pair+num_Flush+num_Four_of_a_kind+num_Full_house+num_Royal_flush+num_Straight+num_Straight_flush+num_Three_of_a_kind+num_Two_pairs+num_nothing_in_hand))*100))
print()

# Seleção de variáveis preditoras (Feature Selection)
#atributos = ['suit_1','rank_1','suit_2','rank_2','suit_3','rank_3','suit_4','rank_4','suit_5','rank_5']
atributos = ['rank_1','rank_2','rank_3','rank_4','rank_5']
# Variável a ser prevista
atrib_prev = ['class']

#cria objetos
X = df[atributos].values
Y = df[atrib_prev].values

#Definindo a taxa de split
split_test_size = 0.70

# Criando dados de treino e de teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X,Y, test_size=split_test_size, random_state= 42)
print()

#imprime os resultados
print("{0:0.2f}% dados de treino".format((len(X_treino)/len(df.index)) * 100))
print("{0:0.2f}% dados de teste".format((len(X_teste)/len(df.index)) * 100))
print()

#verifica a divisão
# print("Original Nothing in Hand: {0} ({1:0.2f}%)".format(len(df.loc[df['class']==0]),
#                                               (len(df.loc[df['class'] == 0])/len(df.index)*100)))
# print("Original One Pair: {0} ({1:0.2f}%)".format(len(df.loc[df['class']==1]),
#                                               (len(df.loc[df['class'] == 1])/len(df.index)*100)))
# print("Original Two Pairs: {0} ({1:0.2f}%)".format(len(df.loc[df['class']==2]),
#                                               (len(df.loc[df['class'] == 2])/len(df.index)*100)))
# print("Original Three of a kind: {0} ({1:0.2f}%)".format(len(df.loc[df['class']==3]),
#                                               (len(df.loc[df['class'] == 3])/len(df.index)*100)))
# print("Original Straight: {0} ({1:0.2f}%)".format(len(df.loc[df['class']==4]),
#                                               (len(df.loc[df['class'] == 4])/len(df.index)*100)))
# print("Original Flush: {0} ({1:0.2f}%)".format(len(df.loc[df['class']==5]),
#                                               (len(df.loc[df['class'] == 5])/len(df.index)*100)))
# print("Original Full House: {0} ({1:0.2f}%)".format(len(df.loc[df['class']==6]),
#                                               (len(df.loc[df['class'] == 6])/len(df.index)*100)))
# print("Original Four of a kind: {0} ({1:0.2f}%)".format(len(df.loc[df['class']==7]),
#                                               (len(df.loc[df['class'] == 7])/len(df.index)*100)))
# print("Original Straight Flush: {0} ({1:0.2f}%)".format(len(df.loc[df['class']==8]),
#                                               (len(df.loc[df['class'] == 8])/len(df.index)*100)))
# print("Original Royal Flush: {0} ({1:0.2f}%)".format(len(df.loc[df['class']==9]),
#                                               (len(df.loc[df['class'] == 9])/len(df.index)*100)))
#
# print("")
#
# print("Treino Nothing in Hand: {0} ({1:0.2f}%)".format(len(Y_treino[Y_treino[:] == 0]),
#                                                           (len(Y_treino[Y_treino[:] == 0])/len(Y_treino) * 100 )))
# print("Treino One Pair: {0} ({1:0.2f}%)".format(len(Y_treino[Y_treino[:] == 1]),
#                                               (len(Y_treino[Y_treino[:]==1])/len(Y_treino)*100)))
# print("Treino Two Pairs: {0} ({1:0.2f}%)".format(len(Y_treino[Y_treino[:]== 2]),
#                                               (len(Y_treino[Y_treino[:]== 2])/len(Y_treino)*100)))
# print("Treino Three of a kind: {0} ({1:0.2f}%)".format(len(Y_treino[Y_treino[:]==3]),
#                                               (len(Y_treino[Y_treino[:]== 3])/len(Y_treino)*100)))
# print("Treino Straight: {0} ({1:0.2f}%)".format(len(Y_treino[Y_treino[:]== 4]),
#                                               (len(Y_treino[Y_treino[:] == 4])/len(Y_treino)*100)))
# print("Treino Flush: {0} ({1:0.2f}%)".format(len(Y_treino[Y_treino[:] == 5]),
#                                               (len(Y_treino[Y_treino[:] == 5])/len(Y_treino)*100)))
# print("Treino Full House: {0} ({1:0.2f}%)".format(len(Y_treino[Y_treino[:] ==6]),
#                                               (len(Y_treino[Y_treino[:] == 6])/len(Y_treino)*100)))
# print("Treino Four of a kind: {0} ({1:0.2f}%)".format(len(Y_treino[Y_treino[:] == 7]),
#                                               (len(Y_treino[Y_treino[:] == 7])/len(Y_treino)*100)))
# print("Treino Straight Flush: {0} ({1:0.2f}%)".format(len(Y_treino[Y_treino[:] ==8]),
#                                               (len(Y_treino[Y_treino[:] == 8])/len(Y_treino)*100)))
# print("Treino Royal Flush: {0} ({1:0.2f}%)".format(len(Y_treino[Y_treino[:] ==9]),
#                                               (len(Y_treino[Y_treino[:] == 9])/len(Y_treino)*100)))
#
# print("")
#
# print("Test Nothing in Hand: {0} ({1:0.2f}%)".format(len(Y_teste[Y_teste[:] == 0]),
#                                                           (len(Y_teste[Y_teste[:] == 0])/len(Y_teste) * 100 )))
# print("Test One Pair: {0} ({1:0.2f}%)".format(len(Y_teste[Y_teste[:] == 1]),
#                                               (len(Y_teste[Y_teste[:]==1])/len(Y_teste)*100)))
# print("Test Two Pairs: {0} ({1:0.2f}%)".format(len(Y_teste[Y_teste[:]== 2]),
#                                               (len(Y_teste[Y_teste[:]== 2])/len(Y_teste)*100)))
# print("Test Three of a kind: {0} ({1:0.2f}%)".format(len(Y_teste[Y_teste[:] ==3]),
#                                               (len(Y_teste[Y_teste[:]== 3])/len(Y_teste)*100)))
# print("Test Straight: {0} ({1:0.2f}%)".format(len(Y_teste[Y_teste[:] == 4]),
#                                               (len(Y_teste[Y_teste[:] == 4])/len(Y_teste)*100)))
# print("Test Flush: {0} ({1:0.2f}%)".format(len(Y_teste[Y_teste[:] == 5]),
#                                               (len(Y_teste[Y_teste[:] == 5])/len(Y_teste)*100)))
# print("Test Full House: {0} ({1:0.2f}%)".format(len(Y_teste[Y_teste[:] ==6]),
#                                               (len(Y_teste[Y_teste[:] == 6])/len(Y_teste)*100)))
# print("Test Four of a kind: {0} ({1:0.2f}%)".format(len(Y_teste[Y_teste[:] == 7]),
#                                               (len(Y_teste[Y_teste[:] == 7])/len(Y_teste)*100)))
# print("Test Straight Flush: {0} ({1:0.2f}%)".format(len(Y_teste[Y_teste[:] ==8]),
#                                               (len(Y_teste[Y_teste[:] == 8])/len(Y_teste)*100)))
# print("Test Royal Flush: {0} ({1:0.2f}%)".format(len(Y_teste[Y_teste[:] ==9]),
#                                               (len(Y_teste[Y_teste[:] == 9])/len(Y_teste)*100)))
print()
#Verifica se alguma linha está com o valor 0
# print("# Linhas no DataFRame {0}".format(len(df)))
# print("# Linhas missing suit_1: {0}".format(len(df.loc[df['suit_1'] == 0])))
# print("# Linhas missing rank_1: {0}".format(len(df.loc[df['rank_1'] == 0])))
# print("# Linhas missing suit_2: {0}".format(len(df.loc[df['suit_2'] == 0])))
# print("# Linhas missing rank_2: {0}".format(len(df.loc[df['rank_2'] == 0])))
# print("# Linhas missing suit_3: {0}".format(len(df.loc[df['suit_3'] == 0])))
# print("# Linhas missing rank_3: {0}".format(len(df.loc[df['rank_3'] == 0])))
# print("# Linhas missing suit_4: {0}".format(len(df.loc[df['suit_4'] == 0])))
# print("# Linhas missing rank_4: {0}".format(len(df.loc[df['rank_4'] == 0])))
# print("# Linhas missing suit_4: {0}".format(len(df.loc[df['suit_5'] == 0])))
# print("# Linhas missing rank_4: {0}".format(len(df.loc[df['rank_5'] == 0])))
print()

#preenche variáveis = 0 com a média dos valores da coluna
preenche_0 = Imputer(missing_values=0,strategy="mean", axis=0)
X_treino = preenche_0.fit_transform(X_treino)
X_teste = preenche_0.fit_transform(X_teste)


#Treinando o modelo
modelo_v1 = GaussianNB()  # type: GaussianNB
#modelo_v1 = MultinomialNB()
#modelo_v1 = BernoulliNB()
modelo_v1.fit(X_treino, Y_treino.ravel())

#score
# print("Score Gaussian treino: ",modelo_v1.score(X_treino, Y_treino))
# print("Score Gaussian teste: ",modelo_v1.score(X_teste, Y_teste))



#Verificando a exatidão no modelo nos dados de treino
nb_predict_treino = modelo_v1.predict(X_treino)

#dados de exatidão de treino
accuracy_treino = metrics.accuracy_score(Y_treino, nb_predict_treino)
print("Treino: Exatidão(Accuracy):", accuracy_treino * 100)
print()

#Verificando a exatidão no modelo nos dados de teste
nb_predict_teste = modelo_v1.predict(X_teste)

#dados de exatidão de teste
accuracy_teste= metrics.accuracy_score(Y_teste, nb_predict_teste)
print("Teste: Exatidão(Accuracy):", accuracy_teste * 100)
print()
# #métricas de desempenho
# print("Confusion Matrix")
# print("{0}".format(metrics.confusion_matrix(Y_teste, nb_predict_teste,labels=[2,1,0])))
print("")
print("Classification Report")
print(metrics.classification_report(Y_treino, nb_predict_treino))
print()
#print("F1_Score Teste: ", metrics.f1_score(Y_teste, nb_predict_teste, average='weighted',labels=np.unique(nb_predict_teste)))
#print("F1_Score Treino: ", metrics.f1_score(Y_treino, nb_predict_treino, average='weighted',labels=np.unique(nb_predict_treino)))
#print("Recall: ",metrics.recall_score(Y_teste, nb_predict_teste,average='weighted', labels=np.unique(nb_predict_teste)))
#print("Similaridade: ",metrics.adjusted_mutual_info_score(nb_predict_teste,Y_teste))
