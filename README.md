*Preditor de Satisfação de Clientes de E-commerce*

Este projeto consiste em um sistema de Machine Learning (ML) desenvolvido para **prever a satisfação do cliente** de uma plataforma de e-commerce (Olist), utilizando dados de pedidos, produtos, clientes e, crucialmente, o **conteúdo dos comentários das avaliações**. O objetivo é identificar proativamente clientes insatisfeitos, permitindo intervenções e melhorias no serviço.

O projeto é modularizado em três pipelines principais:

1.  **Pipeline de Dados:** Extração, limpeza e transformação dos dados brutos.
2.  **Pipeline de Modelos:** Treinamento, avaliação e seleção do melhor modelo de ML.
3.  **Pipeline de Serviço:** Publica o modelo campeão como uma API, tornando-o acessível para predições em tempo real.

## Origem dos Dados

O dataset utilizado é o **Brazilian E-commerce Public Dataset by Olist**, disponível no Kaggle. Ele contém informações reais de 100 mil pedidos feitos em diversas lojas da Olist.

### Pipeline de Dados

Transformar os **dados brutos** de diversas tabelas relacionais do Olist em um formato padronizado e limpo, pronto para a modelagem. O dataset original é composto por informações detalhadas sobre pedidos, avaliações, itens de pedido, produtos e clientes.

Para a construção do dataset final, as seguintes **variáveis (features)** foram selecionadas e preparadas:

* **`price`**: O valor monetário do produto no pedido.
* **`freight_value`**: O custo do frete associado ao pedido.
* **`customer_state`**: O estado geográfico do cliente, uma variável categórica essencial para entender padrões regionais.
* **`product_category_name`**: A categoria à qual o produto pertence, também uma variável categórica que pode influenciar a satisfação.
* **`tempo_de_entrega_dias`**: Esta é uma **feature criada por engenharia de atributos**, calculada como a diferença em dias entre a data de compra (`order_purchase_timestamp`) e a data efetiva de entrega ao cliente (`order_delivered_customer_date`). É um preditor fundamental, pois atrasos na entrega frequentemente impactam a satisfação.

A **variável alvo**, **`target_satisfeito`**, é uma classificação **binária** que define a satisfação do cliente. Ela foi derivada da `review_score` (nota de 1 a 5) da seguinte forma:

* Clientes com `review_score` igual a **4 ou 5** são classificados como **Satisfeitos (1)**.
* Clientes com `review_score` igual a **1, 2 ou 3** são classificados como **Insatisfeitos (0)**.

Essa transformação define o problema como uma **Classificação Binária**. A pipeline de dados também lida com a união das tabelas, tratamento de dados ausentes e a filtragem de registros para garantir a consistência e relevância dos dados para o modelo.

## Desbalanceamento dos Dados e Solução

Distribuição da variável alvo `target_satisfeito`:

* **Clientes Satisfeitos (1):** 26.564 registros.
* **Clientes Insatisfeitos (0):** 13.600 registros.

Essa diferença caracteriza um **desbalanceamento de classes**, onde a classe "Satisfeito" é a majoritária e a "Insatisfeito" é a minoritária. Em problemas de classificação, modelos podem ter dificuldades em aprender com a classe minoritária se o desbalanceamento não for tratado, levando a previsões enviesadas.

Para mitigar esse problema, foi utilizada a técnica de **pesos de classes (class weights)**. Essa abordagem atribui um peso maior às amostras da classe minoritária durante o treinamento do modelo, forçando-o a dar mais atenção a esses casos. A proporção do peso foi calculada com base na razão do número de amostras entre as classes (`peso_classe_1 = contagem_insatisfeitos / contagem_satisfeitos`).

## Lidando com os Comentários dos Produtos (`review_comment_message`)

Os comentários dos produtos são uma fonte rica de informação textual. Para incorporá-los no modelo, foram realizados os seguintes passos:

1.  **Tratamento de Nulos:** Valores ausentes (`NaN`) nos comentários foram preenchidos com strings vazias para evitar erros durante o processamento de texto.
2.  **Vetorização TF-IDF:** A técnica **TF-IDF (Term Frequency-Inverse Document Frequency)** foi aplicada para converter o texto em uma representação numérica. O `TfidfVectorizer` cria vetores numéricos onde cada dimensão representa a importância de uma palavra no contexto de um documento e de todo o corpus de comentários.
3.  **Stop Words em Português:** Palavras comuns e sem muito significado ("de", "a", "o", "que", "e", etc.), conhecidas como *stop words*, foram removidas do texto antes da vetorização. Para isso, foi utilizada a lista de *stop words* para o idioma **português** fornecida pela biblioteca `NLTK`. Isso ajuda o modelo a focar nas palavras mais relevantes para a satisfação do cliente.


## Pipelines Detalhadas do Projeto

### Pipeline de Modelos: Treinamento Inteligente e Seleção Criteriosa

Nesta fase, o foco é construir e validar o modelo preditivo.

* **Preparação dos Dados:** O dataset processado é carregado e dividido em conjuntos de treino e teste, mantendo a proporção das classes (estratificação).
* **Pré-processamento das Features (`make_column_transformer`):** Um transformador de colunas aplica pré-processamentos específicos:
    * **Variáveis Categóricas:** `OneHotEncoder` para codificação.
    * **Variável Textual (`review_comment_message`):** `TfidfVectorizer` com remoção de *stop words* em português.
    * **Variáveis Numéricas:** `StandardScaler` para padronização.
* **Experimentação com Modelos:** Uma coleção de modelos candidatos (Regressão Logística, Random Forest, LightGBM, XGBoost) é treinada dentro de pipelines do Scikit-learn, garantindo que o pré-processamento seja aplicado consistentemente. Todos os modelos incorporam os pesos de classes para lidar com o desbalanceamento.
* **Métrica de Avaliação:** O **F1-Score ponderado** é utilizado como métrica principal para comparar o desempenho dos modelos, sendo ideal para datasets desbalanceados. Relatórios de classificação e matrizes de confusão são gerados para cada modelo.
* **Seleção e Persistência do Modelo Campeão:** O modelo com o melhor F1-Score ponderado é selecionado como o campeão. O **pipeline completo do modelo campeão** (incluindo o pré-processador e o modelo treinado) é salvo no formato `.joblib`, permitindo sua fácil reutilização.

### Pipeline de Serviço: Deploy e Acessibilidade

Esta pipeline transforma o modelo treinado em um serviço web interativo usando **FastAPI**, permitindo que outras aplicações consumam suas predições em tempo real.

* **Inicialização da API:** A API é configurada com `FastAPI`, incluindo título, descrição e versão. Um evento de `startup` tenta abrir automaticamente a documentação interativa (Swagger UI) no navegador.
* **Carregamento do Modelo:** O pipeline do modelo campeão (`modelo_campeao.joblib`) é carregado uma única vez na inicialização do serviço para otimizar a performance, com tratamento de erros para garantir a robustez.
* **Definição dos Modelos de Dados (Pydantic):** `OrderFeatures` e `PredictionOut` definem a estrutura dos dados de entrada e saída, garantindo validação e geração automática de documentação.
* **Definição dos Endpoints da API:**
    * **`/` (GET):** Serve um arquivo `index.html` para uma interface de usuário básica.
    * **`/options` (GET):** Fornece listas únicas de estados e categorias de produtos, extraídas do dataset processado, para preenchimento de formulários em interfaces.
    * **`/predict` (POST):** O endpoint principal. Recebe os dados de um pedido (incluindo o comentário textual), processa-os através do pipeline do modelo e retorna a predição de satisfação (Satisfeito/Insatisfeito).


## Como executar
* Python 3.8 ou superior

Instale as dependencias:

* pip install -r requirements.txt

Execute a main principal:

* python main.py
ou 
* Run and Debug --> Executar Pipeline Completa
