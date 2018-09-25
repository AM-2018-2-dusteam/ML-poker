# ML-poker

## Sobre

...

## Organização dos dados

Os dados contidos no csv estão divididos em dados de treino e de test, e neles são estão contidos os naipes(suit) e valores(rank) para uma mão de cinco cartas e no final a classificação da mão.

### Dados da carta

* ***naipes - suit***

  |Valor noarquivo|Significado|
  |---------------|-----------|
  |1|Copas|
  |2|Espadas|
  |3|Ouro|
  |4|Paus|

* ***valores - rank***

  |Valor noarquivo|Significado|
  |---------------|-----------|
  |1|A|
  |2|2|
  |3|3|
  |4|4|
  |5|5|
  |6|6|
  |7|7|
  |8|8|
  |9|9|
  |10|10|
  |11|Valete|
  |12|Dama|
  |13|Rei|
  
  * ***Classificação da mão***
 

  |Valor noarquivo|Significado|Ocorrencias nos dados de treino|Ocorrencia nos dados de teste|
  |---------------|-----------|-----------|-------------------------------------------------|
  |0|Nada na mão|12493|501209|
  |1|Um par|10599|422498|
  |2|Dois pares|1206|47622|
  |3|Um trio|513|21121|
  |4|Straight|93|3885|
  |5|Flush|54|1996|
  |6|Full house|36|1424|
  |7|Uma quadra|6|230|
  |8|Straight flush|5|12|
  |9|Royal flush|5|3|
  
  
  ***Tamanho dados de teste: 1,000,000***
  ***Tamanho dados de treino: 25010***
  


### Colunas 
