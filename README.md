# MiniProjeto_Modulo2_IA-1_SCTEC
MiniProjeto_Modulo2_IA#1_SCTEC

Objetivo:  construir, treinar, validar, comparar e estressar um Pipeline Preditivo Multiclasse ponta a ponta capaz de classificar dígitos manuscritos a partir do dataset MNIST (mnist_784).

# Fase 1: Carregamento e Análise Exploratória de Imagens (EDA)
Importar o dataset MNIST via scikit-learn (fetch_openml('mnist_784')) ou via tensorflow.keras.datasets.mnist.
Exibir a dimensionalidade das matrizes (X e y) e comprovar a distribuição das classes (balanceamento de dígitos de 0 a 9).
Gerar uma grade visual  de no mínimo 2 x 5 com matplotlib, exibindo exemplos de imagens de cada dígito junto com seus respectivos rótulos originais.
Inserir célula de texto interpretando a estrutura dos dados: explicar o significado da escala de intensidade de pixels (0 a 255) e como imagens bidimensionais (28 x 28) são representadas vetorialmente (784 features).

# Fase 2: Pipeline de Pré-processamento e Divisão dos Dados
Divisão Estratificada: Dividir a base em Treino, Validação e Teste (ex: 70% treino, 10% validação e 20% teste ou 80% treino/validação e 20% teste), garantindo estratificação por classe (stratify=y).
Normalização / Escalonamento: Aplicar o redimensionamento dos pixels para a escala [0.0, 1.0] (dividindo por 255.0 ou via MinMaxScaler/StandardScaler).
Justificar textualmente a importância da normalização para a convergência de modelos lineares e distâncias métricas.

# Fase 3: Implementação e Treinamento dos 3 Modelos
Deve-se obrigatoriamente escolher e treinar 3 modelos distintos, podendo optar por combinações de:
Modelos Clássicos: SVM (Support Vector Machines), Random Forest, KNN (K-Nearest Neighbors), XGBoost / Gradient Boosting, Regressão Logística Multinomial, etc.
Redes Neurais: Perceptron Multicamadas (MLP via Scikit-Learn ou TensorFlow/Keras).
Cada modelo deve possuir ajuste justificado de ao menos 2 hiperparâmetros (ex: n_estimators e max_depth na Random Forest; n_neighbors e weights no KNN; C e kernel no SVM; número de neurônios, taxa de aprendizado ou funções de ativação na Rede Neural).
Caso queiram usar outros modelos não citados acima, é permitido, desde que justificadas as escolhas.

# Fase 4: Avaliação Comparativa de Desempenho
Para cada um dos 3 modelos avaliados no conjunto de teste independente, apresentar:
Matriz de Confusão completa (10 x 10) com mapa de calor (heatmap).
Tabela comparativa consolidada contendo:
Acurácia Global (Accuracy)
Precisão Média Ponderada (Precision)
Revocação/Sensibilidade Média Ponderada (Recall)
F1-Score Ponderado (F1-Score)
Sugestão: classification_report
Célula de conclusão técnica apontando: qual dígito apresentou a maior taxa de confusão (ex: 4 vs 9, 3 vs 5, 7 vs 1), qual modelo obteve a melhor performance e qual o impacto do custo computacional (tempo de treino vs acurácia).

## Fase 5: Nos próximos sub itens da Fase 5, vocês irão testar a capacidade dos modelos de lidar com cenários fora do padrão ideal de laboratório:
# Fase 5.1 - Desafio (A) Treinamento Restrito com Classes Ocultadas (Class Masking)
Selecione ao menos duas classes para serem totalmente removidas da base de treinamento (por exemplo: ocultar os dígitos 4 e 7).
Treine um dos seus modelos (clássico ou rede neural) sem nunca ter visto esses dois dígitos durante o ajuste dos pesos.

# Fase 5.2 - Desafio (B) Teste de Generalização Extrema (Inferência OOD)
Submeta o modelo treinado no Desafio A a um conjunto de teste contendo exclusivamente as classes que foram ocultadas (no exemplo: apenas imagens dos dígitos 4 e 7).
Análise e plote o comportamento do modelo:
Como o classificador reage ao ser forçado a classificar algo que nunca viu?
Quais classes conhecidas ele atribui a esses dígitos desconhecidos?
Exiba a matriz de confusão correspondente, apresente suas impressões sobre o teste e discuta o conceito de "falsa certeza" (overconfidence) em modelos de IA.

# Fase 5.3 - Desafio (C) Inferência com Imagens Manuscritas Próprias
Deve-se escrever manualmente um dígito em folha de papel branco com caneta escura (ou desenhar no Paint/GIMP com fundo preto e traço branco). 
Note que dependendo de sua escolha o pipeline de pré-processamento vai ser diferente.
Digitalize / fotografe as imagens e construa um pipeline de pré-processamento em Python (usando por exemplo PIL/OpenCV) contendo:
Conversão para escala de cinza;
Inversão de cores (caso o papel seja branco com traço preto);
Redimensionamento para 28 x 28 pixels com centralização de massa/bounding box;
Normalização para o intervalo [0.0, 1.0].
Realize a predição das suas imagens próprias utilizando o melhor modelo desenvolvido e plote a imagem processada ao lado do gráfico de probabilidades de saída.
# MiniProjeto_Modulo2_IA-1_SCTEC
MiniProjeto_Modulo2_IA#1_SCTEC
