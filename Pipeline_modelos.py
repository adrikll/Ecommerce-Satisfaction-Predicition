# Importação de bibliotecas
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Ferramentas do Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer # Para transformar texto em features numéricas (NLP)
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Modelos candidatos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Importações para tratar o desbalanceamento de classes
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline # Pipeline especial que permite reamostragem

def run_model_pipeline():
    print("Iniciando Módulo de Pipeline de Modelos (Versão com NLP e SMOTE)...")

    # --- FASE 1: PREPARAÇÃO DOS DADOS ---
    processed_data_path = os.path.join("output", "dados_processados.csv")
    try:
        df = pd.read_csv(processed_data_path)
        # Garante que a coluna de comentários seja tratada como string e que
        # valores nulos sejam convertidos para texto vazio.
        df['review_comment_message'] = df['review_comment_message'].astype(str).fillna('')
    except FileNotFoundError:
        print(f"Erro: Arquivo '{processed_data_path}' não encontrado.")
        print("Por favor, execute o pipeline de dados atualizado primeiro.")
        return

    print("Dados carregados com sucesso.")
    df['target_satisfeito'] = df['review_score'].apply(lambda x: 1 if x >= 4 else 0)
    
    X = df.drop(['review_score', 'target_satisfeito'], axis=1)
    y = df['target_satisfeito']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nDados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

    # Racional: O pré-processador  tratar as variáveis categóricas
    # e processa a coluna de texto.
    categorical_features = ['customer_state', 'product_category_name']
    text_feature = 'review_comment_message'
    
    preprocessor = make_column_transformer(
        # 1. Transforma as colunas categóricas em variáveis numéricas.
        (OneHotEncoder(handle_unknown='ignore'), categorical_features),
        
        # 2. Transforma o texto dos comentários em um vetor numérico.
        # TfidfVectorizer mede a importância de cada palavra.
        # max_features=500 limita o vocabulário às 500 palavras mais relevantes.
        (TfidfVectorizer(max_features=500, stop_words='english'), text_feature),
        
        # 3. Mantém as colunas restantes (numéricas) como estão.
        remainder='passthrough'
    )

    # --- FASE 2: EXPERIMENTAÇÃO COM MODELOS E SMOTE ---
    # Racional: Para combater o desbalanceamento, usamos o SMOTE. Ele deve ser
    # aplicado APENAS nos dados de treino para evitar vazamento de dados.
    # O `make_imb_pipeline` garante que isso aconteça automaticamente durante o
    # treinamento (`.fit()`).
    models = {
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        "LightGBM": LGBMClassifier(random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    results = {}
    print("\nIniciando experimentação com modelos candidatos usando SMOTE...")

    for model_name, model in models.items():
        # Cria um pipeline especial que primeiro aplica o pré-processamento,
        # depois o SMOTE para reamostrar os dados, e por fim treina o modelo.
        pipeline = make_imb_pipeline(
            preprocessor,
            SMOTE(random_state=42),
            model
        )
        
        print(f"--- Treinando {model_name} ---")
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        # Avalia o modelo usando o F1-Score ponderado.
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[model_name] = {'f1_score': f1, 'pipeline': pipeline}
        print(f"F1-Score Ponderado do {model_name}: {f1:.4f}")

    # --- FASE 3: SELEÇÃO DO CAMPEÃO PELO F1-SCORE ---
    # Racional: A acurácia pode ser enganadora em datasets desbalanceados.
    # O F1-Score é uma média harmônica entre precisão e recall, sendo uma
    # métrica muito mais confiável para selecionar o melhor modelo geral.
    champion_model_name = max(results, key=lambda k: results[k]['f1_score'])
    champion_pipeline = results[champion_model_name]['pipeline']
    champion_f1 = results[champion_model_name]['f1_score']

    print("-" * 50)
    print(f"🏆 Modelo Campeão: {champion_model_name} com F1-Score de {champion_f1:.4f}")
    print("-" * 50)

    print("Gerando relatório de classificação final para o modelo campeão...")
    y_pred_champion = champion_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred_champion, target_names=['Insatisfeito (0)', 'Satisfeito (1)'])
    
    print("\nRelatório de Classificação Detalhado (Modelo Campeão):")
    print(report)
    
    # Gerando e salvando a Matriz de Confusão
    print("Gerando Matriz de Confusão...")
    cm = confusion_matrix(y_test, y_pred_champion)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Insatisfeito (0)', 'Satisfeito (1)'], yticklabels=['Insatisfeito (0)', 'Satisfeito (1)'])
    plt.xlabel('Previsto'); plt.ylabel('Verdadeiro'); plt.title(f'Matriz de Confusão - {champion_model_name}')
    
    output_dir = "output"; os.makedirs(output_dir, exist_ok=True)
    confusion_matrix_path = os.path.join(output_dir, "matriz_confusao_campeao.png")
    plt.savefig(confusion_matrix_path)
    print(f"Matriz de confusão salva em: {confusion_matrix_path}")
    plt.show()

    # --- FASE 4: GERAÇÃO DO BINÁRIO ---
    model_path = os.path.join(output_dir, "modelo_campeao.joblib")
    joblib.dump(champion_pipeline, model_path)
    print(f"\nModelo campeão salvo com sucesso em: {model_path}")

if __name__ == "__main__":
    run_model_pipeline()
