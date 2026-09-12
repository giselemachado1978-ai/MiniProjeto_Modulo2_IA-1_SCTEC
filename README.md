# MiniProjeto_Modulo2_IA-1_SCTEC
MiniProjeto_Modulo2_IA#1_SCTEC

Objetivo:  construir, treinar, validar, comparar e estressar um Pipeline Preditivo Multiclasse ponta a ponta capaz de classificar dígitos manuscritos a partir do dataset MNIST (mnist_784).

# Fase 1: Carregamento e Análise Exploratória de Imagens (EDA)
Importa o dataset MNIST via scikit-learn (fetch_openml('mnist_784')).
Exibir a dimensionalidade das matrizes (X e y) e comprovar a distribuição das classes (balanceamento de dígitos de 0 a 9).
Gerar uma grade visual  de 2 x 5 com matplotlib, exibindo exemplos de imagens de cada dígito junto com seus respectivos rótulos originais.
Apresenta descrição/explicação da estrutura dos dados: explica o significado da escala de intensidade de pixels (0 a 255) e como imagens bidimensionais (28 x 28) são representadas vetorialmente (784 features).

# Fase 2: Pipeline de Pré-processamento e Divisão dos Dados
Divisão Estratificada: Divide a base em Treino, Validação e Teste (80% treino/validação e 20% teste, depois, 70% treino, 10% validação), estratificando por classe (stratify=y).
Normalização / Escalonamento: Aplica o redimensionamento dos pixels para a escala [0.0, 1.0] (dividindo por 255.0).
Apresenta texto justificando a importância da normalização para a convergência de modelos lineares e distâncias métricas.

# Fase 3: Implementação e Treinamento dos 3 Modelos
Escolhidos para treino os modelos : KNN, Random Forest e Regressao Logistica
obs.: Em principio, tentei utilizar o Perceptron no lugar do KNN, mas meu sistema de poucos recursos, nao rodou. 
Cada modelo foi ajustado com 2 hiperparâmetros (n_estimators e max_depth na Random Forest; n_neighbors e weights no KNN; C e solver na regressao).

# Fase 4: Avaliação Comparativa de Desempenho
Para cada um dos 3 modelos apresentou-se:
Matriz de Confusão completa (10 x 10) com mapa de calor (heatmap).
  Uso do classification_report para apresentar os dados:
  Acurácia Global (Accuracy)
  Precisão Média Ponderada (Precision)
  Revocação/Sensibilidade Média Ponderada (Recall)
  F1-Score Ponderado (F1-Score)
E por fim, apresentou-se uma tabela comparativa consolidada e conclusoes acerca de qual dígito apresentou a maior taxa de confusão, qual modelo obteve a melhor performance e qual o impacto do custo computacional (tempo de treino vs acurácia).

## Fase 5: Teste de capacidade dos modelos de lidar com cenários fora do padrão ideal de laboratório:
# Fase 5.1 - Desafio (A) Treinamento Restrito com Classes Ocultadas (Class Masking)
Foram removidas da base de treino os digitos 4 e 7.
Executado treino do modelo Random Forest sem esses dígitos.

# Fase 5.2 - Desafio (B) Teste de Generalização Extrema (Inferência OOD)
Aqui, o modelo treinado no Desafio A é submetido a um conjunto de teste contendo apenas os digitos que foram ocultadas antes.
Apresenta a matriz de confusão neste novo cenário;
O texto apresenta conclusões acerca do comportamento do modelo quanto às questoes abaixo e uma discussão sobre o conceito de "falsa certeza" (overconfidence) em modelos de IA:
  Como o classificador reage ao ser forçado a classificar algo que nunca viu?
  Quais classes conhecidas ele atribui a esses dígitos desconhecidos?

# Fase 5.3 - Desafio (C) Inferência com Imagens Manuscritas Próprias
Criado o digito "8" no Paint 3D, branco com fundo preto e salvo como o arquivo "Feitoamao.png".
Utilizou-se o PIL para o pré-processamento da imagem (conversao de escala, redimensionamento, centralização/bounding box e normalização/escalonamento).
Então usou-se o modelo KNN para prever a classe da minha imagem e o resultado é apresentado no gráfico de probabilidade, **mostrando que o modelo não é o mais adequado ao problema, técnicas de ODD poderiam ser aplicadas pra evitar essa provável "overconfidence"**. 
