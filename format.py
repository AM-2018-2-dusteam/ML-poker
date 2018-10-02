import pandas as pd
import numpy as np

def format_poker_data(X):

    X = pd.get_dummies(X,columns=['suit_1','suit_2','suit_3','suit_4','suit_5'])

    rank, suit = np.split(X,[5],axis=1)

    rank.values.sort()

    suit = suit.astype('Bool')

    copas = suit['suit_1_1'] & suit['suit_2_1'] & suit['suit_3_1'] &suit['suit_4_1'] &suit['suit_5_1']
    espadas = suit['suit_1_2'] & suit['suit_2_2'] & suit['suit_3_2'] &suit['suit_4_2'] &suit['suit_5_2']
    ouro = suit['suit_1_3'] & suit['suit_2_3'] & suit['suit_3_3'] &suit['suit_4_3'] &suit['suit_5_3']
    paus = suit['suit_1_4'] & suit['suit_2_4'] & suit['suit_3_4'] &suit['suit_4_4'] &suit['suit_5_4']

    all_equal = copas | espadas | ouro | paus

    # Retorna um DataFrame com os valores das vartas e um vetor booleano indicando quando a mão é so de uma cor.
    return (rank, all_equal)