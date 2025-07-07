import kagglehub
import pandas as pd
import os

def run_data_pipeline():
    print("Iniciando Módulo de Pipeline de Dados...")
    print("Baixando os dados do Kaggle...")
    try:
        path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
        print(f"Download concluído. Arquivos em: {path}")
    except Exception as e:
        print(f"Erro no download: {e}")
        return

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

    # 3. Combinação dos dados (Merge)
    print("Combinando os datasets...")
    df = pd.merge(orders, reviews, on="order_id")
    df = pd.merge(df, order_items, on="order_id")
    df = pd.merge(df, products, on="product_id")
    df = pd.merge(df, customers, on="customer_id")
    
    # 4. Limpeza e Tratamento dos dados
    print("Iniciando limpeza e tratamento...")
    
    cols_to_use = [
        'order_id',
        'review_score',
        'price',
        'freight_value',
        'customer_state',
        'product_category_name',
        'order_status',
        'order_purchase_timestamp',
        'order_delivered_customer_date'
    ]
    df = df[cols_to_use]

    df.dropna(subset=['order_delivered_customer_date', 'product_category_name'], inplace=True)
    
    date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    df['tempo_de_entrega_dias'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

    df = df[df['tempo_de_entrega_dias'] >= 0]
    df.dropna(subset=['tempo_de_entrega_dias'], inplace=True)
    
    final_cols = [
        'review_score',
        'price',
        'freight_value',
        'customer_state',
        'product_category_name',
        'tempo_de_entrega_dias'
    ]
    df = df[final_cols]
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dados_processados.csv")
    df.to_csv(output_path, index=False)
    
    print("-" * 50)
    print(f"Pipeline de dados concluído com sucesso!")
    print(f"Arquivo processado salvo em: {output_path}")
    print(f"Total de registros no arquivo final: {len(df)}")
    print("-" * 50)
    
if __name__ == "__main__":
    run_data_pipeline()