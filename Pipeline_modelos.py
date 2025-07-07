import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import joblib
import os

def run_model_pipeline():
    print("Iniciando Módulo de Pipeline de Modelos...")

    processed_data_path = os.path.join("output", "dados_processados.csv")
    try:
        df = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{processed_data_path}' não encontrado.")
        print("Por favor, execute o script 'pipeline_dados.py' primeiro.")
        return

    print("Dados carregados com sucesso.")
    
    X = df.drop('review_score', axis=1)
    y = df['review_score']
    categorical_features = ['customer_state', 'product_category_name']
    
   
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), categorical_features),
        remainder='passthrough'
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")
    
    model = make_pipeline(
    preprocessor,
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    )
    
    print("Iniciando o treinamento do modelo...")
    model.fit(X_train, y_train)
    print("Treinamento concluído.")
    
    print("Avaliando o modelo no conjunto de teste...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("-" * 50)
    print(f"Acurácia do Modelo: {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(report)
    print("-" * 50)
    
    output_dir = "output"
    model_path = os.path.join(output_dir, "modelo_campeao.joblib")
    joblib.dump(model, model_path)
    
    print(f"Modelo salvo com sucesso em: {model_path}")

if __name__ == "__main__":
    run_model_pipeline()