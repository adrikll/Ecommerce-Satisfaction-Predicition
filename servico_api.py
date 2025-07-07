import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="API de Predição de Avaliações de Pedidos",
    description="Serviço que utiliza um modelo de Machine Learning para prever a nota de avaliação (1 a 5) de um pedido.",
    version="1.0.0"
)

try:
    model = joblib.load("output/modelo_campeao.joblib")
    print("Modelo carregado com sucesso.")
except FileNotFoundError:
    print("Erro: Arquivo do modelo 'modelo_campeao.joblib' não encontrado.")
    model = None

class OrderFeatures(BaseModel):
    price: float
    freight_value: float
    customer_state: str
    product_category_name: str
    tempo_de_entrega_dias: int
    
    class Config:
        schema_extra = {
            "example": {
                "price": 129.90,
                "freight_value": 22.50,
                "customer_state": "SP",
                "product_category_name": "cama_mesa_banho",
                "tempo_de_entrega_dias": 10
            }
        }

@app.post("/predict")
def predict(features: OrderFeatures):
    if model is None:
        return {"error": "Modelo não foi carregado. A predição não pode ser realizada."}
        
    input_data = pd.DataFrame([features.dict()])
    prediction = model.predict(input_data)
    
    return {"previsao_nota_avaliacao": int(prediction[0])}

@app.get("/")
def read_root():
    return {"status": "API online. Acesse /docs para ver a documentação interativa."}