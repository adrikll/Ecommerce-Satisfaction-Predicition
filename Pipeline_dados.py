import kagglehub  #para interagir com o Kaggle Datasets Hub
import pandas as pd #para manipulação e análise de dados
import os         #para interações com o sistema operacional (criar pastas e caminhos)

def run_data_pipeline():
    """
    Função principal que orquestra todo o pipeline de dados:
    1. Extração (Download dos dados)
    2. Carga Inicial (Leitura dos arquivos CSV)
    3. Transformação (Merge, Limpeza, Engenharia de Atributos)
    4. Carregamento (Salvamento do arquivo processado)
    """
    print("Iniciando Módulo de Pipeline de Dados...")
    
    '''
    EXTRAÇÃO
      Racional: O primeiro passo é obter os dados brutos da fonte.
      Usamos o kagglehub para baixar programaticamente o dataset, garantindo
      reprodutibilidade e evitando o download manual. Um bloco try-except
      captura possíveis falhas de conexão ou autenticação.
    '''
    
    print("Baixando os dados do Kaggle (olistbr/brazilian-ecommerce)...")
    try:
        #baixa o dataset e retorna o caminho para o diretório local
        path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
        print(f"Download concluído. Arquivos estão em: {path}")
    except Exception as e:
        print(f"Erro crítico no download: {e}")
        return
    
    '''
    CARGA INICIAL
    Racional: Carregamos os datasets essenciais para o problema em DataFrames
    do pandas. A seleção dos arquivos é baseada na necessidade de conectar
    informações do pedido, cliente, produto, itens do pedido e avaliação.
    '''
    print("Carregando datasets principais...")
    try:
        orders = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"))
        reviews = pd.read_csv(os.path.join(path, "olist_order_reviews_dataset.csv"))
        order_items = pd.read_csv(os.path.join(path, "olist_order_items_dataset.csv"))
        products = pd.read_csv(os.path.join(path, "olist_products_dataset.csv"))
        customers = pd.read_csv(os.path.join(path, "olist_customers_dataset.csv"))
    except FileNotFoundError as e:
        print(f"Erro ao carregar arquivo: {e}. Verifique o caminho e o resultado do download.")
        return

    '''
    TRANSFORMAÇÃO
    Racional: Esta é a fase central, onde os dados brutos são convertidos
    em informações úteis e de alta qualidade.
    '''
    # --------------------------------------------------------------------------
    '''
    Combinação dos dados (Merge)
    Racional: Os dados estão em formato relacional (normalizado). Para análise,
    precisamos de uma visão unificada (desnormalizada). Unimos os DataFrames
    usando chaves comuns (order_id, product_id, customer_id) para criar um
    único dataset que conecta cada item de pedido à sua avaliação, produto,
    cliente e detalhes da entrega.
    '''
    
    print("Combinando os datasets...")
    df = pd.merge(orders, reviews, on="order_id")
    df = pd.merge(df, order_items, on="order_id")
    df = pd.merge(df, products, on="product_id")
    df = pd.merge(df, customers, on="customer_id")
    
    '''
    Seleção de Variáveis
    Racional: Selecionamos apenas as colunas relevantes para o nosso problema
    (prever review_score), descartando o resto para simplificar o modelo e
    reduzir o ruído. A escolha de cada coluna é justificada abaixo:
        - order_id: Chave de identificação temporária.
        - review_score: Nossa variável-alvo (target). É o que queremos prever.
        - price / freight_value: Variáveis preditoras. O preço do produto e do frete podem 
        influenciar a percepção de valor do cliente.
        - customer_state: Variável preditora. A localização do cliente pode impactar o tempo
        de entrega e, consequentemente, a satisfação.
        - product_category_name: Variável preditora. A categoria do produto pode ter diferentes
        níveis de satisfação esperados.
        - order_status: Usado para filtrar apenas pedidos concluídos.
        - order_purchase_timestamp / order_delivered_customer_date: Necessários para calcular 
        o tempo de entrega.
    '''
    print("Selecionando colunas de interesse...")
    cols_to_use = [
        'order_id', 'review_score', 'price', 'freight_value', 'customer_state',
        'product_category_name', 'order_status', 'order_purchase_timestamp',
        'order_delivered_customer_date', 'review_comment_message' 
    ]
    df = df[cols_to_use]

    #Limpeza e Tratamento dos dados
    print("Iniciando limpeza e tratamento...")
    
    '''
    Racional: Removemos linhas duplicadas do DataFrame. Isso garante que cada
    registro seja único e evita que o modelo seja treinado com dados redundantes.
    '''
    df.drop_duplicates(inplace=True)
    
    '''
      Racional: Para o nosso modelo, só fazem sentido pedidos que foram efetivamente
      entregues, pois a experiência de entrega é um fator chave da satisfação.
      Filtramos apenas os pedidos com status 'delivered'.
    '''
    df = df[df['order_status'] == 'delivered'].copy()
    
    '''
    Racional: Para garantir a integridade do dataset e evitar erros futuros,
    removemos todas as linhas com valores nulos. Isso é uma abordagem segura,
    especialmente quando a quantidade de dados nulos é pequena.
    '''
    df.dropna(inplace=True)
    
    '''
    Racional: As colunas de data vêm como texto (object). É fundamental
    convertê-las para o tipo datetime para realizar cálculos de tempo.
    'errors=coerce' transforma datas inválidas em NaT (Not a Time), que podem
    ser tratadas posteriormente.
    '''
    
    date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    '''
    Engenharia de Atributos e Criação da Label
    Racional: Criamos uma nova variável, 'tempo_de_entrega_dias', que é um
    preditor muito mais poderoso do que as datas brutas. A hipótese é que
    tempos de entrega mais longos levam a avaliações piores.
    '''
    df['tempo_de_entrega_dias'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

    '''
    Racional: Após o cálculo, podem surgir valores inválidos.
        - Tempo de entrega negativo: Indica um erro nos dados (entrega antes da compra).
          Esses registros são inconsistentes e devem ser removidos.
        - Tempo de entrega nulo (NaT): Resultante da conversão de datas inválidas.
          Também devem ser removidos;
    '''
    
    df = df[df['tempo_de_entrega_dias'] >= 0]
    df.dropna(subset=['tempo_de_entrega_dias'], inplace=True)
    df['tempo_de_entrega_dias'] = df['tempo_de_entrega_dias'].astype(int)
    
    '''
    Criação da Variável Alvo
    Racional: A variável-alvo 'target_satisfeito' é criada a partir da
    'review_score'. Consideramos clientes satisfeitos aqueles com nota 4 ou 5
    (label 1) e insatisfeitos os com nota 1, 2 ou 3 (label 0).
    Isso transforma nosso problema de regressão (prever nota) em um problema
    de classificação binária (prever satisfação).
    '''
    df['target_satisfeito'] = df['review_score'].apply(lambda x: 1 if x >= 4 else 0)
    
    '''
    Seleção Final de Colunas
    Racional: Com as features criadas e os dados limpos, selecionamos o conjunto
    final de colunas que serão salvas. Removemos colunas intermediárias que
    não serão usadas diretamente no modelo (como as datas originais).
    '''
    
    final_cols = [
        'target_satisfeito', 'review_score', 'price', 'freight_value', 'customer_state',
        'product_category_name', 'tempo_de_entrega_dias', 'review_comment_message'
    ]
    df = df[final_cols]
    
    '''
    CARREGAMENTO
    Racional: O passo final é persistir o dataset processado. Salvamos em um
    novo arquivo CSV dentro de um diretório 'output'. Isso mantém o projeto
    organizado e separa os dados brutos dos dados tratados.
    '''
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True) #cria o diretório se não existir
    output_path = os.path.join(output_dir, "dados_processados.csv")
    df.to_csv(output_path, index=False)
    
    print("-" * 50)
    print(f"Pipeline de dados concluído com sucesso!")
    print(f"Arquivo processado salvo em: {output_path}")
    print(f"O dataset final contém {len(df)} registros e {len(df.columns)} colunas.")
    print("-" * 50)
    
if __name__ == "__main__":
    run_data_pipeline()