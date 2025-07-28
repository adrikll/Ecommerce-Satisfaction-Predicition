import subprocess
import time
import os
import sys

from Pipeline_dados import run_data_pipeline
from Pipeline_modelos import run_model_pipeline

def run_full_pipeline():
    """
    Orquestra a execução completa do pipeline de dados, modelagem e inicia a API.
    """
    print("Iniciando a execução completa da pipeline...\n")

    #Executar Pipeline de Dados
    print("--- Passo 1: Executando o Pipeline de Dados ---")
    run_data_pipeline()
    print("--- Pipeline de Dados Concluído ---\n")

    processed_data_path = os.path.join("output", "dados_processados.csv")
    if not os.path.exists(processed_data_path):
        print(f"Erro: O arquivo '{processed_data_path}' não foi encontrado. "
              "O pipeline de dados pode ter falhado. Abortando a execução.")
        return

    #Executar Pipeline de Modelagem
    print("--- Passo 2: Executando o Pipeline de Modelagem ---")
    run_model_pipeline()
    print("--- Pipeline de Modelagem Concluído ---\n")

    #Verifica se o modelo foi salvo antes de iniciar a API
    model_path = os.path.join("output", "modelo_campeao.joblib")
    if not os.path.exists(model_path):
        print(f"Erro: O arquivo do modelo '{model_path}' não foi encontrado. "
              "O pipeline de modelagem pode ter falhado. Abortando a execução da API.")
        return

    #Iniciar API e Interface
    print("--- Passo 3: Iniciando o Serviço da API ---")
    print("A API será iniciada em um novo processo. Pressione Ctrl+C para encerrá-la.")
    
    # Comando para iniciar o Uvicorn
    # Usa sys.executable para garantir que o mesmo interpretador Python seja usado
    command = [sys.executable, "-m", "uvicorn", "servico_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    
    #Inicia o processo da API em segundo plano
    #Usamos Popen para não bloquear o script principal
    api_process = None
    try:
        api_process = subprocess.Popen(command, cwd=os.path.dirname(os.path.abspath(__file__)))
        print(f"API iniciada com PID: {api_process.pid}")
        print("Aguarde alguns segundos para a API inicializar e abrir no navegador (se configurado).")
        
        # Mantém o script principal rodando enquanto a API estiver ativa
        # O usuário precisará encerrar manualmente (Ctrl+C no terminal)
        # para parar a API.
        api_process.wait() # Espera o processo da API terminar

    except FileNotFoundError:
        print("Erro: 'uvicorn' não encontrado. Certifique-se de que está instalado e no seu PATH.")
    except Exception as e:
        print(f"Erro ao iniciar a API: {e}")
    finally:
        if api_process and api_process.poll() is None: # Verifica se o processo ainda está rodando
            print("Encerrando o processo da API...")
            api_process.terminate() # Encerra o processo da API
            api_process.wait(timeout=5) # Aguarda o processo terminar
            if api_process.poll() is None: # Se ainda não terminou, mata
                api_process.kill()
            print("Processo da API encerrado.")

    print("\nExecução completa da pipeline finalizada.")

if __name__ == "__main__":
    run_full_pipeline()
