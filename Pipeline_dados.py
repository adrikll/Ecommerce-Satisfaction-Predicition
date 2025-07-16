# Importação de bibliotecas essenciais
import kagglehub  # Para interagir com o Kaggle Datasets Hub
import pandas as pd # Biblioteca fundamental para manipulação e análise de dados
import os         # Para interações com o sistema operacional, como criar pastas e caminhos

def run_data_pipeline():
    """
    Função principal que orquestra todo o pipeline de dados:
    1. Extração (Download dos dados)
    2. Carga Inicial (Leitura dos arquivos CSV)
    3. Transformação (Merge, Limpeza, Engenharia de Atributos)
    4. Carregamento (Salvamento do arquivo processado)
    """
    print("Iniciando Módulo de Pipeline de Dados...")
    
    # --------------------------------------------------------------------------
    # FASE 1: EXTRAÇÃO (Extract)
    # Racional: O primeiro passo é obter os dados brutos da fonte.
    # Usamos o kagglehub para baixar programaticamente o dataset, garantindo
    # reprodutibilidade e evitando o download manual. Um bloco try-except
    # captura possíveis falhas de conexão ou autenticação.
    # --------------------------------------------------------------------------
    print("Baixando os dados do Kaggle (olistbr/brazilian-ecommerce)...")
    try:
        # Baixa o dataset e retorna o caminho para o diretório local
        path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
        print(f"Download concluído. Arquivos estão em: {path}")
    except Exception as e:
        print(f"Erro crítico no download: {e}")
        return

    # --------------------------------------------------------------------------
    # FASE 2: CARGA INICIAL (Initial Load)
    # Racional: Carregamos os datasets essenciais para o problema em DataFrames
    # do pandas. A seleção dos arquivos é baseada na necessidade de conectar
    # informações do pedido, cliente, produto, itens do pedido e avaliação.
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # FASE 3: TRANSFORMAÇÃO (Transform)
    # Racional: Esta é a fase central, onde os dados brutos são convertidos
    # em informações úteis e de alta qualidade.
    # --------------------------------------------------------------------------

    # 3.1. Combinação dos dados (Merge)
    # Racional: Os dados estão em formato relacional (normalizado). Para análise,
    # precisamos de uma visão unificada (desnormalizada). Unimos os DataFrames
    # usando chaves comuns (order_id, product_id, customer_id) para criar um
    # único dataset que conecta cada item de pedido à sua avaliação, produto,
    # cliente e detalhes da entrega.
    print("Combinando os datasets...")
    df = pd.merge(orders, reviews, on="order_id")
    df = pd.merge(df, order_items, on="order_id")
    df = pd.merge(df, products, on="product_id")
    df = pd.merge(df, customers, on="customer_id")
    
    # 3.2. Seleção de Variáveis (Feature Selection)
    # Racional: Selecionamos apenas as colunas relevantes para o nosso problema
    # (prever review_score), descartando o resto para simplificar o modelo e
    # reduzir o ruído. A escolha de cada coluna é justificada abaixo:
    #   - order_id: Chave de identificação temporária.
    #   - review_score: Nossa variável-alvo (target). É o que queremos prever.
    #   - price / freight_value: Variáveis preditoras. O preço do produto e do frete podem influenciar a percepção de valor do cliente.
    #   - customer_state: Variável preditora. A localização do cliente pode impactar o tempo de entrega e, consequentemente, a satisfação.
    #   - product_category_name: Variável preditora. A categoria do produto pode ter diferentes níveis de satisfação esperados.
    #   - order_status: Usado para filtrar apenas pedidos concluídos.
    #   - order_purchase_timestamp / order_delivered_customer_date: Necessários para calcular o tempo de entrega.
    print("Selecionando colunas de interesse...")
    cols_to_use = [
        'order_id', 'review_score', 'price', 'freight_value', 'customer_state',
        'product_category_name', 'order_status', 'order_purchase_timestamp',
        'order_delivered_customer_date'
    ]
    df = df[cols_to_use]

    # 3.3. Limpeza e Tratamento dos dados (Data Cleaning)
    print("Iniciando limpeza e tratamento...")

    # Racional: Para o nosso modelo, só fazem sentido pedidos que foram efetivamente
    # entregues, pois a experiência de entrega é um fator chave da satisfação.
    # Filtramos apenas os pedidos com status 'delivered'.
    df = df[df['order_status'] == 'delivered'].copy()
    
    # Racional: Se a data de entrega ou a categoria do produto estiverem ausentes,
    # não podemos calcular features importantes (tempo de entrega) ou usar uma
    # variável preditora chave (categoria). Como a quantidade de nulos é pequena
    # em relação ao total, a remoção é uma estratégia segura e simples.
    df.dropna(subset=['order_delivered_customer_date', 'product_category_name'], inplace=True)
    
    # Racional: As colunas de data vêm como texto (object). É fundamental
    # convertê-las para o tipo datetime para realizar cálculos de tempo.
    # 'errors=coerce' transforma datas inválidas em NaT (Not a Time), que podem
    # ser tratadas posteriormente.
    date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # 3.4. Engenharia de Atributos (Feature Engineering)
    # Racional: Criamos uma nova variável, 'tempo_de_entrega_dias', que é um
    # preditor muito mais poderoso do que as datas brutas. A hipótese é que
    # tempos de entrega mais longos levam a avaliações piores.
    df['tempo_de_entrega_dias'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

    # Racional: Após o cálculo, podem surgir valores inválidos.
    #   - Tempo de entrega negativo: Indica um erro nos dados (entrega antes da compra).
    #     Esses registros são inconsistentes e devem ser removidos.
    #   - Tempo de entrega nulo (NaT): Resultante da conversão de datas inválidas.
    #     Também devem ser removidos.
    df = df[df['tempo_de_entrega_dias'] >= 0]
    df.dropna(subset=['tempo_de_entrega_dias'], inplace=True)
    df['tempo_de_entrega_dias'] = df['tempo_de_entrega_dias'].astype(int)
    
    # 3.5. Seleção Final de Colunas
    # Racional: Com as features criadas e os dados limpos, selecionamos o conjunto
    # final de colunas que serão salvas. Removemos colunas intermediárias que
    # não serão usadas diretamente no modelo (como as datas originais).
    final_cols = [
        'review_score', 'price', 'freight_value', 'customer_state',
        'product_category_name', 'tempo_de_entrega_dias'
    ]
    df = df[final_cols]
    
    # --------------------------------------------------------------------------
    # FASE 4: CARREGAMENTO (Load)
    # Racional: O passo final é persistir o dataset processado. Salvamos em um
    # novo arquivo CSV dentro de um diretório 'output'. Isso mantém o projeto
    # organizado e separa os dados brutos dos dados tratados.
    # --------------------------------------------------------------------------
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True) # Cria o diretório se não existir
    output_path = os.path.join(output_dir, "dados_processados.csv")
    df.to_csv(output_path, index=False)
    
    print("-" * 50)
    print(f"Pipeline de dados concluído com sucesso!")
    print(f"Arquivo processado salvo em: {output_path}")
    print(f"O dataset final contém {len(df)} registros e {len(df.columns)} colunas.")
    print("-" * 50)
    
if __name__ == "__main__":
    run_data_pipeline()