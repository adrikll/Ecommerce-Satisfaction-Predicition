# Importação de bibliotecas
import pandas as pd
import os
import joblib

# Ferramentas do Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Modelos candidatos atualizados
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier  # XGBoost

# Métricas de avaliação
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def run_model_pipeline():
    """
    Função que orquestra todo o pipeline de modelagem com as seguintes melhorias:
    1. Reenquadramento do problema para classificação binária (Satisfeito vs. Insatisfeito).
    2. Adição do XGBoost ao conjunto de modelos candidatos.
    3. Avaliação mais detalhada com matriz de confusão.
    """
    print("Iniciando Módulo de Pipeline de Modelos (Versão Otimizada)...")

    # --------------------------------------------------------------------------
    # FASE 1: PREPARAÇÃO DOS DADOS
    # --------------------------------------------------------------------------
    processed_data_path = os.path.join("output", "dados_processados.csv")
    try:
        df = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{processed_data_path}' não encontrado.")
        print("Por favor, execute o pipeline de dados primeiro.")
        return

    print("Dados carregados com sucesso.")

    # --- ETAPA CHAVE: REENQUADRAMENTO DO PROBLEMA ---
    # Racional: Como discutido, o problema original de 5 classes era muito
    # desbalanceado. Transformá-lo em um problema binário é a estratégia mais
    # eficaz para obter um modelo útil e com melhor performance.
    print("\nReenquadrando o problema para classificação binária...")
    # Satisfeito (1) = nota 4 ou 5
    # Insatisfeito (0) = nota 1, 2 ou 3
    df['target_satisfeito'] = df['review_score'].apply(lambda x: 1 if x >= 4 else 0)

    print("Distribuição da nova variável alvo (0=Insatisfeito, 1=Satisfeito):")
    print(df['target_satisfeito'].value_counts(normalize=True))

    # X agora usa todas as colunas exceto a original e a nova alvo
    X = df.drop(['review_score', 'target_satisfeito'], axis=1)
    y = df['target_satisfeito']  # y é a nossa nova variável alvo binária

    # Divisão em treino e teste (estratificando pela nova 'y')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nDados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

    # Definição do pré-processador (sem alterações)
    categorical_features = ['customer_state', 'product_category_name']
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        remainder='passthrough'
    )

    # --------------------------------------------------------------------------
    # FASE 2: EXPERIMENTAÇÃO COM MODELOS CANDIDATOS (PROBLEMA BINÁRIO)
    # Racional: Testamos um conjunto de modelos poderosos, incluindo o XGBoost.
    # Nota: Não precisamos mais do 'class_weight' pois o problema está mais balanceado.
    # --------------------------------------------------------------------------
    models = {
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "LightGBM": LGBMClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    results = {}
    print("\nIniciando experimentação com modelos candidatos...")

    for model_name, model in models.items():
        pipeline = make_pipeline(preprocessor, model)
        print(f"--- Treinando {model_name} ---")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = {'accuracy': accuracy, 'pipeline': pipeline}
        print(f"Acurácia do {model_name}: {accuracy:.4f}")

    # --------------------------------------------------------------------------
    # FASE 3: SELEÇÃO E AVALIAÇÃO DETALHADA DO MODELO CAMPEÃO
    # --------------------------------------------------------------------------
    champion_model_name = max(results, key=lambda k: results[k]['accuracy'])
    champion_pipeline = results[champion_model_name]['pipeline']
    champion_accuracy = results[champion_model_name]['accuracy']

    print("-" * 50)
    print(f"🏆 Modelo Campeão: {champion_model_name} com acurácia de {champion_accuracy:.4f}")
    print("-" * 50)

    print("Gerando relatório de classificação final para o modelo campeão...")
    y_pred_champion = champion_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred_champion, target_names=['Insatisfeito (0)', 'Satisfeito (1)'])
    
    print("\nRelatório de Classificação Detalhado (Modelo Campeão):")
    print(report)
    
    # Gerando Matriz de Confusão para uma análise visual
    print("Gerando Matriz de Confusão...")
    cm = confusion_matrix(y_test, y_pred_champion)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Insatisfeito (0)', 'Satisfeito (1)'], 
                yticklabels=['Insatisfeito (0)', 'Satisfeito (1)'])
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.title(f'Matriz de Confusão - {champion_model_name}')
    
    # Salvando a imagem da matriz de confusão
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    confusion_matrix_path = os.path.join(output_dir, "matriz_confusao_campeao.png")
    plt.savefig(confusion_matrix_path)
    print(f"Matriz de confusão salva em: {confusion_matrix_path}")
    plt.show()

    # --------------------------------------------------------------------------
    # FASE 4: GERAÇÃO DO BINÁRIO (CARGA)
    # --------------------------------------------------------------------------
    model_path = os.path.join(output_dir, "modelo_campeao.joblib")
    joblib.dump(champion_pipeline, model_path)
    print(f"\nModelo campeão salvo com sucesso em: {model_path}")


if __name__ == "__main__":
    run_model_pipeline()