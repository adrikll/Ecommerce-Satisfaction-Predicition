import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="API de Predição de Avaliação de Livros",
    description="Serviço que utiliza um modelo otimizado para prever a nota (1 a 5) de um livro.",
    version="2.0.0"
)

try:
    model = joblib.load("output/modelo_livros_otimizado.joblib")
    categorias_raras = [
        'Add a comment', 'Cultural', 'Erotica', 'Humor', 'Manga', 'Parenting',
        'Philosophy', 'Psychology', 'Religion', 'Self Help', 'Short Stories',
        'Spirituality', 'Womens Fiction'
    ]
except FileNotFoundError:
    model = None
    categorias_raras = []

class BookFeatures(BaseModel):
    preco: float
    categoria: str
    titulo: str
    
    class Config:
        schema_extra = {
            "example": {
                "preco": 25.50,
                "categoria": "History",
                "titulo": "The Black Maria"
            }
        }

@app.post("/predict_rating")
def predict_rating(features: BookFeatures):
    if model is None:
        return {"error": "Modelo não carregado."}
    
    input_df = pd.DataFrame([features.dict()])
    
    input_df['comprimento_titulo'] = input_df['titulo'].apply(len)
    
    if input_df['categoria'].iloc[0] in categorias_raras:
        input_df['categoria_agrupada'] = 'Outros'
    else:
        input_df['categoria_agrupada'] = input_df['categoria']
    
    final_input_df = input_df[['preco', 'categoria_agrupada', 'comprimento_titulo']]
    
    prediction = model.predict(final_input_df)
    
    return {
        "previsao_nota_avaliacao": int(prediction[0]),
    }

@app.get("/")
def read_root():
    return {"status": "API de Predição de Livros (Otimizada) Online. Acesse /docs."}