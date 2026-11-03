# 🌱 Análise e Previsão da Produção de Sementes em Santa Catarina

## Aplicação

🔗 **Acesse o aplicativo:** [Análise de Sementes - Santa Catarina](https://arthur-gruber-analise-sementes-sc-app-yolzgm.streamlit.app/)

---

## 2. Análise Exploratória de Dados (EDA)

Nesta etapa, investigamos o conjunto de dados filtrado especificamente para o estado de Santa Catarina (SC), visando compreender a distribuição das áreas de produção de sementes, as espécies mais cultivadas e a relação entre área e produção.

### 2.1. Estrutura dos Dados

Após o carregamento e a limpeza inicial (tratamento de valores nulos e conversão de tipos), o dataset de Santa Catarina apresentou a seguinte estrutura:

- **Total de registros:** 55.449 campos de produção.
- **Granularidade:** cada linha representa um campo de produção de sementes (Inscrição).
- **Período:** dados históricos desde a safra 2013/2013 até as previsões de safras futuras (2025/2026).

### 2.2. Estatística Descritiva

Foram analisadas as medidas de tendência central e dispersão das principais variáveis numéricas: Área (ha) e Produção Estimada.

| Métrica | Área (hectares) | Produção Estimada (kg/ton) |
|---|---:|---:|
| Contagem | 55.449 | 55.449 |
| Média | 31,29 | 238,63 |
| Mediana | 20,00 | 82,80 |
| Desvio padrão | 37,00 | 4.514,35 |
| Mínimo | 0,00 | 0,00 |
| Máximo | 800,00 | 360.000,00 |

**Principais insights:**

- **Heterogeneidade:** o desvio padrão da Área (37,00) é maior que a própria média (31,29), indicando que Santa Catarina possui desde pequenos produtores até grandes propriedades, com concentração em áreas menores.
- **Disparidade na produção:** a média de produção é de 238,63 unidades, enquanto o valor máximo chega a 360.000. Isso sugere a presença de outliers extremos ou diferenças de escala de medida entre as culturas.
- **Mediana vs. média:** a mediana da área (20 ha) é inferior à média (31 ha), confirmando uma distribuição assimétrica à direita, com muitos registros de pequenas áreas e poucos registros de áreas muito grandes.

### 2.3. Visualizações e Gráficos

Foram gerados três gráficos principais para facilitar a interpretação dos dados.

**A) Distribuição da Área Plantada (Histograma)**

- **Objetivo:** apresentar a frequência de campos de produção por tamanho de área.
- **Análise:** foi necessário utilizar uma escala logarítmica devido à grande concentração de registros em áreas pequenas (menores que 50 ha). Isso reforça o perfil da agricultura familiar e das pequenas propriedades em Santa Catarina.

**B) Top 10 Espécies Produzidas**

- **Objetivo:** identificar as espécies com maior número de registros de campos de sementes em SC.
- **Análise:** o gráfico permite identificar quais culturas predominam no cenário catarinense, como *Glycine max* (Soja), *Zea mays* (Milho) e *Oryza sativa* (Arroz). Essa identificação é importante porque cada espécie apresenta uma produtividade esperada diferente.

**C) Correlação: Área x Produção (Scatter Plot)**

- **Objetivo:** demonstrar a relação entre o tamanho da área e a produção estimada.
- **Análise:** observou-se uma tendência positiva: quanto maior a área, maior tende a ser a produção. Entretanto, a distribuição dos pontos apresenta grande dispersão.
- **Motivo:** diferentes espécies possuem produtividades distintas. Um hectare de fumo, por exemplo, apresenta um volume de produção diferente de um hectare de soja. A mistura dessas espécies no mesmo gráfico gera ruído visual e estatístico.

### 2.4. Correlação entre Variáveis

A análise de correlação, posteriormente complementada pela importância das variáveis no modelo Random Forest, indicou que:

- **Área:** variável mais fortemente correlacionada com a Produção Estimada.
- **Categoria e Espécie:** atuam como fatores modificadores, pois a relação entre área e produção varia conforme a espécie e a categoria da semente.

---

## 3. Pré-processamento de Dados

Nesta etapa, preparamos o dataset para a modelagem, priorizando a qualidade dos dados e a conversão para formatos numéricos adequados.

### 3.1. Limpeza e Conversão de Tipos

Os dados originais do governo brasileiro utilizam vírgulas para separar casas decimais e pontos para milhares (exemplo: `1.234,56`). Esse formato é interpretado pelo Python como texto (`string`).

**Ações realizadas:**

- Remoção dos pontos utilizados como separadores de milhares.
- Substituição das vírgulas decimais por pontos.
- Conversão dos valores para tipos numéricos.

**Variáveis afetadas:**
- `Area`
- `Producao estimada`
- `Producao bruta`

**Conversão de datas:** a coluna `Data do Plantio` foi convertida de texto para o formato `datetime`, permitindo a extração de informações temporais, caso necessário.

### 3.2. Tratamento de Valores Ausentes (Missing Values)

A presença de valores nulos pode impedir o funcionamento de diversos algoritmos de Machine Learning.

**Análise:** observamos que a coluna `Producao bruta` possuía muitos valores ausentes, pois nem todos os campos já haviam sido colhidos. Já a coluna `Producao estimada` estava preenchida na maioria dos registros.

**Ação:** optamos por remover as linhas que continham valores nulos nas colunas selecionadas para o modelo:

- `Area`
- `Especie`
- `Categoria`
- `Ano_Safra`
- `Producao estimada`

Essa abordagem garantiu que o modelo fosse treinado apenas com registros completos nas variáveis selecionadas.

### 3.3. Engenharia de Atributos (Feature Engineering)

Criamos uma nova variável a partir das informações existentes para facilitar o aprendizado do modelo.

**Nova variável:** `Ano_Safra`

- **Origem:** coluna `Safra`, originalmente no formato `2013/2013`.
- **Processo:** extração do primeiro ano numérico da safra.
- **Resultado:** conversão do ano para um número inteiro sequencial, permitindo que o modelo considere a evolução temporal.

### 3.4. Codificação de Variáveis Categóricas (Encoding)

Os modelos matemáticos trabalham com valores numéricos, sendo necessário transformar variáveis textuais em números.

**Técnica utilizada:** Label Encoding.

**Aplicação:**

- `Especie`: cada nome de planta recebeu um número único (exemplo: Soja = 15, Milho = 28).
- `Categoria`: as categorias de sementes (S1, S2, C1, C2) foram transformadas em números sequenciais.

**Justificativa:** o Label Encoding é eficiente para variáveis com muitas categorias distintas, como é o caso das espécies, evitando a criação de centenas de novas colunas que poderiam surgir com o One-Hot Encoding.

### 3.5. Seleção de Variáveis (Feature Selection)

Foram selecionadas apenas as colunas relevantes para a previsão, descartando identificadores administrativos, como `Inscrição`, `Município` e `Cultivar`, que poderiam gerar ruído ou overfitting.

**Variáveis de entrada (X - Features):**

- `Area`
- `Especie_Encoded`
- `Categoria_Encoded`
- `Ano_Safra`

**Variável de saída (y - Target):**

- `Producao estimada`

### 3.6. Normalização e Escalonamento

**Decisão:** não foi aplicado escalonamento, como `StandardScaler` ou `MinMaxScaler`.

**Justificativa técnica:** o algoritmo escolhido, Random Forest, é baseado em árvores de decisão. Esses algoritmos não são sensíveis à escala dos dados, pois trabalham com regras de divisão, como `Área > 50`, independentemente da amplitude dos valores.

Portanto, a normalização não traria ganho significativo de desempenho, e manter os dados na escala original facilita a interpretação dos resultados.

### 3.7. Divisão em Treino e Teste

Para validar o modelo, os dados foram separados em dois grupos:

| Conjunto | Proporção | Finalidade |
|---|---:|---|
| Treino | 80% | Dados utilizados para o aprendizado do modelo. |
| Teste | 20% | Dados inéditos utilizados para avaliar o desempenho final. |

---

## 4. Modelagem de Dados

Nesta etapa, construímos e treinamos o modelo preditivo para estimar a produção de sementes em Santa Catarina.

O objetivo foi resolver um problema de **Regressão**, no qual a variável alvo (*target*) é a `Producao estimada`.

### 4.1. Seleção do Algoritmo

Optamos pelo algoritmo **Random Forest Regressor (Floresta Aleatória)**.

**Justificativas técnicas:**

- **Robustez:** o Random Forest é um método Ensemble que combina múltiplas árvores de decisão e calcula a média de suas previsões. Isso reduz o risco de overfitting em comparação com uma única árvore de decisão.
- **Dados mistos:** o dataset contém variáveis numéricas (`Area`) e categóricas transformadas (`Especie` e `Categoria`). Métodos baseados em árvores conseguem trabalhar com essa combinação sem exigir normalização complexa.
- **Não linearidade:** a relação entre área e produção agrícola não é perfeitamente linear. O Random Forest consegue capturar padrões complexos e não lineares melhor que uma Regressão Linear em determinadas situações.

### 4.2. Configuração do Experimento

Utilizamos a biblioteca **Scikit-Learn (Python)** com a seguinte configuração:

**Divisão dos dados:**

- **Treino (80%):** utilizado para o ajuste do modelo.
- **Teste (20%):** utilizado exclusivamente para a avaliação final, garantindo que o modelo seja testado com dados que não foram utilizados durante o treinamento.

**Métricas de avaliação:**

- RMSE (*Root Mean Squared Error*).
- R² (*R-Squared*).

### 4.3. Otimização de Hiperparâmetros (Tuning)

Para evitar a utilização exclusiva das configurações padrão (*default*), aplicamos uma busca aleatória de melhores parâmetros por meio do `RandomizedSearchCV`.

**Parâmetros testados:**

| Parâmetro | Valores testados | Descrição |
|---|---|---|
| `n_estimators` | 50, 100, 200 | Número de árvores utilizadas. |
| `max_depth` | None, 10, 20, 30 | Profundidade máxima das árvores. |
| `min_samples_split` | 2, 5, 10 | Número mínimo de amostras necessário para dividir um nó. |

**Melhor configuração encontrada:**

| Parâmetro | Melhor valor |
|---|---:|
| `n_estimators` | 100 |
| `max_depth` | 10 |
| `min_samples_split` | 10 |

### 4.4. Resultados e Interpretação

O desempenho do modelo nos dados de teste de Santa Catarina foi:

| Métrica | Resultado |
|---|---:|
| RMSE | 3.145,53 |
| R² Score | -0,09 |

**RMSE (Raiz do Erro Quadrático Médio): 3.145,53**

O resultado indica uma magnitude de erro de aproximadamente 3.145 kg nas previsões, considerando a escala utilizada na variável de produção.

Para produções de centenas de toneladas, esse erro pode ser considerado razoável. Entretanto, para pequenas produções, representa uma diferença significativa.

**R² Score (Coeficiente de Determinação): -0,09**

O valor próximo de zero e levemente negativo indica que o modelo apresentou dificuldade para explicar a variância dos dados.

**Análise crítica:**

O principal fator identificado foi a alta heterogeneidade das culturas agrícolas. O objetivo inicial foi construir um modelo generalista capaz de prever a produção de todas as espécies de Santa Catarina simultaneamente.

Entretanto, diferentes culturas apresentam dinâmicas de produção muito distintas.

Por exemplo, a dinâmica de produção da soja é drasticamente diferente daquela encontrada em hortaliças, tanto em volume quanto em produtividade por área.

Ao combinar essas diferentes características em um único modelo, o algoritmo encontrou dificuldades para estabelecer regras generalizadas de previsão.

### 4.5. Importância das Variáveis (Feature Importance)

O algoritmo também permitiu identificar quais variáveis tiveram maior influência nas previsões.

| Variável | Importância aproximada |
|---|---:|
| Área | 0,58 |
| Categoria | 0,15 |
| Espécie | 0,14 |

**Interpretação:**

- **Área (~0,58):** principal fator de influência. Quanto maior a área, maior tende a ser a produção.
- **Categoria (~0,15):** a categoria da semente (Básica, C1, C2, S1, S2) influencia a produtividade esperada.
- **Espécie (~0,14):** o tipo de planta também apresenta influência relevante nas previsões.

---

## 5. Ajuste de Hiperparâmetros (Hyperparameter Tuning)

Para otimizar o desempenho do modelo `RandomForestRegressor`, realizamos um processo de ajuste fino dos hiperparâmetros.

O objetivo foi encontrar um equilíbrio entre viés (*underfitting*) e variância (*overfitting*), buscando melhorar a capacidade de generalização do modelo para novos dados.

### 5.1. Metodologia de Busca

Utilizamos a técnica **Randomized Search Cross Validation (`RandomizedSearchCV`)**.

**Por que escolhemos essa técnica?**

Diferentemente do Grid Search, que testa todas as combinações possíveis e pode apresentar um custo computacional elevado, a busca aleatória seleciona diferentes combinações de parâmetros.

Isso permite explorar um espaço maior de configurações de maneira mais eficiente.

**Validação Cruzada (Cross-Validation):**

Definimos `cv=3`. O conjunto de treinamento foi dividido em três partes, realizando três etapas de treinamento e validação para cada configuração testada, buscando maior robustez estatística.

### 5.2. Espaço de Parâmetros Explorados

Foram definidos os seguintes parâmetros para a otimização:

**`n_estimators` (Número de Árvores)**

- Valores testados: `[50, 100, 200]`.
- Hipótese: um número maior de árvores tende a estabilizar as previsões, mas também aumenta o custo computacional.

**`max_depth` (Profundidade Máxima)**

- Valores testados: `[None, 10, 20, 30]`.
- Hipótese: árvores muito profundas podem memorizar ruídos dos dados (*overfitting*). Limitar a profundidade força o modelo a aprender padrões mais gerais.

**`min_samples_split` (Mínimo de Amostras para Divisão)**

- Valores testados: `[2, 5, 10]`.
- Hipótese: exigir mais amostras para criar um novo ramo evita regras específicas demais para poucos registros isolados.

### 5.3. Resultados Obtidos

Após a execução do algoritmo de busca, a melhor combinação de parâmetros encontrada para os dados de Santa Catarina foi:

| Parâmetro | Melhor valor |
|---|---:|
| `n_estimators` | 100 |
| `max_depth` | 10 |
| `min_samples_split` | 10 |

### 5.4. Interpretação da Escolha

A escolha automática desses parâmetros permite identificar algumas características importantes dos dados:

**Profundidade Limitada (`max_depth=10`)**

O modelo selecionou uma profundidade de 10, em vez de 30 ou ilimitada. Isso indica que limitar a profundidade pode ajudar a reduzir a influência de ruídos e outliers extremos, favorecendo o aprendizado de padrões mais gerais.

**Regularização (`min_samples_split=10`)**

O modelo selecionou um valor superior ao padrão (2), indicando uma configuração mais conservadora, que exige um grupo maior de amostras para realizar novas subdivisões. Isso ajuda a reduzir o overfitting, especialmente em espécies com poucos registros.

**Estabilidade (`n_estimators=100`)**

A configuração com 100 árvores apresentou o melhor resultado encontrado. O aumento para 200 árvores não trouxe ganho significativo que justificasse o custo computacional adicional.

### 5.5. Impacto no Desempenho Final

Com os hiperparâmetros ajustados, o modelo alcançou os seguintes resultados:

- **RMSE:** 3.145,53.
- **R² Score:** -0,09.

Esses resultados representam o desempenho obtido para o conjunto de dados utilizado, considerando as limitações relacionadas à heterogeneidade das culturas agrícolas.

A análise também evidencia a possibilidade de aprimoramento futuro por meio da criação de modelos específicos para diferentes espécies, evitando combinar culturas com características de produção muito distintas.

---

## 6. Considerações Finais

O projeto permitiu explorar os dados de produção de sementes de Santa Catarina, identificar padrões relevantes e desenvolver um modelo de Machine Learning para estimar a produção agrícola.

Durante o processo, foram realizadas etapas de:

- Análise Exploratória de Dados (EDA).
- Limpeza e pré-processamento dos dados.
- Engenharia e seleção de atributos (*Feature Engineering* e *Feature Selection*).
- Codificação de variáveis categóricas.
- Treinamento de modelo de regressão utilizando Random Forest.
- Otimização de hiperparâmetros com `RandomizedSearchCV`.
- Avaliação de desempenho utilizando RMSE e R².
- Análise da importância das variáveis (*Feature Importance*).

Embora o modelo tenha apresentado limitações na capacidade de generalização, principalmente devido à diversidade das culturas agrícolas, os resultados permitiram identificar a área de produção como a variável de maior influência nas estimativas.

O estudo também aponta oportunidades de melhoria, especialmente por meio da construção de modelos especializados por espécie agrícola.
