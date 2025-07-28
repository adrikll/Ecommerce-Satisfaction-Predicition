import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import webbrowser
from fastapi.responses import FileResponse, JSONResponse 

#INICIALIZAÇÃO DA API 
app = FastAPI(
    title="API de Predição de Satisfação do Cliente",
    description="Serviço que utiliza um modelo de Machine Learning para prever a satisfação do cliente, incluindo análise de comentários.",
    version="2.3.0" # Nova versão com suporte a comentários
)

#Função para abrir o navegador 
@app.on_event("startup")
def open_browser_on_startup():
    try:
        webbrowser.open("http://127.0.0.1:8000")
    except Exception as e:
        print(f"Não foi possível abrir o navegador automaticamente: {e}")

#CARREGAMENTO DO MODELO 
try:
    model = joblib.load("output/modelo_campeao.joblib")
    print("Modelo carregado com sucesso.")
except FileNotFoundError:
    print("Erro: Arquivo do modelo não encontrado.")
    model = None
except Exception as e:
    print(f"Ocorreu um erro ao carregar o modelo: {e}")
    model = None

#DEFINIÇÃO DOS MODELOS DE DADOS 
class OrderFeatures(BaseModel):
    price: float = Field(..., example=129.90)
    freight_value: float = Field(..., example=22.50)
    customer_state: str = Field(..., example="SP")
    product_category_name: str = Field(..., example="cama_mesa_banho")
    tempo_de_entrega_dias: int = Field(..., example=10)
    review_comment_message: str = Field("", example="Gostei muito do produto, entrega rápida!")

class PredictionOut(BaseModel):
    classe_predita: int = Field(..., example=1)
    previsao: str = Field(..., example="Satisfeito")

#DEFINIÇÃO DOS ENDPOINTS DA API
@app.get("/", response_class=FileResponse)
def read_root():
    """Serve a interface do usuário (arquivo index.html)."""
    return FileResponse('index.html')

@app.get("/options")
def get_options():
    """
    Lê o arquivo de dados processados e retorna listas únicas de estados
    e categorias de produtos para preencher os dropdowns da interface.
    """
    try:
        df = pd.read_csv("output/dados_processados.csv")
        states = sorted(list(df['customer_state'].unique()))
        categories = sorted(list(df['product_category_name'].unique()))
        
        return JSONResponse(content={
            "states": states,
            "categories": categories
        })
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Arquivo 'dados_processados.csv' não encontrado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler as opções: {e}")


@app.post("/predict", response_model=PredictionOut)
def predict(features: OrderFeatures):
    """
    Recebe os dados de um pedido, incluindo o comentário, e retorna a predição de satisfação.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não está disponível.")
    try:
        
        input_data = pd.DataFrame([features.dict()])
        
        prediction_class = model.predict(input_data)[0]
        prediction_label = "Satisfeito" if prediction_class == 1 else "Insatisfeito"
        return PredictionOut(classe_predita=int(prediction_class), previsao=prediction_label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro durante a predição: {e}")