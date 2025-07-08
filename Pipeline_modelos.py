import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import os

df = pd.read_csv(os.path.join("output", "livros_extraidos.csv"))
df.dropna(inplace=True)

df['comprimento_titulo'] = df['titulo'].apply(len)

limite_contagem = 10
contagem_categorias = df['categoria'].value_counts()
categorias_raras = contagem_categorias[contagem_categorias < limite_contagem].index
df['categoria_agrupada'] = df['categoria'].replace(categorias_raras, 'Outros')

X = df[['preco', 'categoria_agrupada', 'comprimento_titulo']]
y = df['nota_avaliacao']

preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), ['categoria_agrupada']),
    (StandardScaler(), ['preco', 'comprimento_titulo']),
    remainder='passthrough'
)

pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(random_state=42, class_weight='balanced')
)

param_grid = {
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_depth': [10, 20, None],
    'randomforestclassifier__min_samples_split': [2, 5]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
print(f"Melhores Parâmetros: {grid_search.best_params_}")
print(f"Melhor Acurácia (Validação Cruzada): {grid_search.best_score_:.4f}")

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "modelo_livros_otimizado.joblib")
joblib.dump(best_model, model_path)

print(f"Modelo otimizado salvo em: {model_path}")