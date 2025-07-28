import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#ferramentas do Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

# modelos candidatos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def run_model_pipeline():
    """
    Função principal que orquestra o pipeline de modelos:
    1. Carrega e prepara os dados processados.
    2. Divide os dados em conjuntos de treino e teste.
    3. Define um pré-processador para transformar as features.
    4. Configura e treina modelos de classificação, usando uma estratégia
       unificada de pesos de classes para tratar o desbalanceamento.
    5. Avalia, compara e seleciona o melhor modelo com base no F1-Score.
    6. Salva os resultados de cada modelo e o modelo campeão.
    """
    print("Iniciando Módulo de Pipeline de Modelos (Versão Unificada de Pesos de Classes)...")

    #PREPARAÇÃO DOS DADOS
    processed_data_path = os.path.join("output", "dados_processados.csv")
    try:
        df = pd.read_csv(processed_data_path)
        #Garante que a coluna de comentários seja tratada como string.
        df['review_comment_message'] = df['review_comment_message'].astype(str).fillna('')
    except FileNotFoundError:
        print(f"Erro: Arquivo '{processed_data_path}' não encontrado.")
        print("Por favor, execute o pipeline de dados atualizado primeiro.")
        return

    print("Dados carregados com sucesso.")
    
    #separação das features (X) e da variável-alvo (y)
    X = df.drop(['target_satisfeito', 'review_score'], axis=1)
    y = df['target_satisfeito']

    #contagem de classes para calcular os pesos
    class_counts = y.value_counts()
    neg, pos = class_counts[0], class_counts[1]
    
    print("\nContagem de classes na variável-alvo (target_satisfeito):")
    print(f"Insatisfeito (0): {neg}")
    print(f"Satisfeito (1): {pos}")
    
    #define os pesos das classes para os modelos.
    scale_pos_weight = neg / pos
    class_weight_dict = {0: 1, 1: scale_pos_weight}
    print(f"\nPeso para a classe 1 (Satisfeito) em relação à classe 0 (Insatisfeito): {scale_pos_weight:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nDados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

    #PRÉ-PROCESSAMENTO DAS FEATURES
    categorical_features = ['customer_state', 'product_category_name']
    text_feature = 'review_comment_message'
    numerical_features = ['price', 'freight_value', 'tempo_de_entrega_dias']
    
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), categorical_features),
        (TfidfVectorizer(max_features=500, stop_words='english'), text_feature),
        (StandardScaler(), numerical_features),
        remainder='passthrough'
    )

    #EXPERIMENTAÇÃO COM MODELOS
    models = {
        "Regressão Logística": LogisticRegression(max_iter=5000, random_state=42, class_weight=class_weight_dict),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight=class_weight_dict),
        "LightGBM": LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    }

    results = {}
    
    print("\nIniciando experimentação com modelos candidatos usando pesos de classes...")

    for model_name, model in models.items():
        print(f"\n--- Treinando {model_name} ---")
        
        #cria um diretório de resultados específico para o modelo
        model_results_dir = os.path.join("output", "model_results", model_name.replace(' ', '_'))
        os.makedirs(model_results_dir, exist_ok=True)
        
        if model_name in ["Regressão Logística", "Random Forest"]:
            print(f"Pesos de classes para {model_name}:")
            print(f"  Classe 0 (Insatisfeito): {class_weight_dict[0]:.2f}")
            print(f"  Classe 1 (Satisfeito): {class_weight_dict[1]:.2f}")
        else: # LightGBM e XGBoost
            print(f"Peso para a classe minoritária (Satisfeito) em {model_name}:")
            print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

        #cria um pipeline padrão que aplica o pré-processamento e treina o modelo.
        pipeline = make_pipeline(preprocessor, model)
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        #avaliação do modelo e salvamento dos resultados
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[model_name] = {'f1_score': f1, 'pipeline': pipeline}
        print(f"F1-Score Ponderado do {model_name}: {f1:.4f}")

        report = classification_report(y_test, y_pred, target_names=['Insatisfeito (0)', 'Satisfeito (1)'])
        report_path = os.path.join(model_results_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Relatório salvo em: {report_path}")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Insatisfeito (0)', 'Satisfeito (1)'], yticklabels=['Insatisfeito (0)', 'Satisfeito (1)'])
        plt.xlabel('Previsto'); plt.ylabel('Verdadeiro'); plt.title(f'Matriz de Confusão - {model_name}')
        confusion_matrix_path = os.path.join(model_results_dir, "confusion_matrix.png")
        plt.savefig(confusion_matrix_path)
        plt.close()
        print(f"Matriz de confusão salva em: {confusion_matrix_path}")
    
    # SELEÇÃO E PERSISTÊNCIA DO MODELO CAMPEÃO
    champion_model_name = max(results, key=lambda k: results[k]['f1_score'])
    champion_pipeline = results[champion_model_name]['pipeline']
    champion_f1 = results[champion_model_name]['f1_score']
    
    print("-" * 50)
    print(f"Modelo Campeão: {champion_model_name} com F1-Score de {champion_f1:.4f}")
    print("-" * 50)

    print("Gerando relatório de classificação final para o modelo campeão...")
    y_pred_champion = champion_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred_champion, target_names=['Insatisfeito (0)', 'Satisfeito (1)'])
    print("\nRelatório de Classificação Detalhado (Modelo Campeão):")
    print(report)

    cm = confusion_matrix(y_test, y_pred_champion)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Insatisfeito (0)', 'Satisfeito (1)'], yticklabels=['Insatisfeito (0)', 'Satisfeito (1)'])
    plt.xlabel('Previsto'); plt.ylabel('Verdadeiro'); plt.title(f'Matriz de Confusão - {champion_model_name} (Campeão)')
    confusion_matrix_path = os.path.join("output", "matriz_confusao_campeao.png")
    plt.savefig(confusion_matrix_path)
    print(f"Matriz de confusão do campeão salva em: {confusion_matrix_path}")

    #GERAÇÃO DO BINÁRIO
    model_path = os.path.join("output", "modelo_campeao.joblib")
    joblib.dump(champion_pipeline, model_path)
    print(f"\nModelo campeão salvo com sucesso em: {model_path}")

if __name__ == "__main__":
    run_model_pipeline()
