🤖 Preditor de Satisfação de Clientes de E-commerce

Um projeto completo de Data Science que constrói e implanta um modelo de Machine Learning para prever a satisfação de clientes, servido através de uma API interativa com interface web.

Status do Projeto: Concluído ✅

A interface web permite que o utilizador insira dados de um pedido e receba uma previsão de satisfação em tempo real.

📝 Visão Geral do Projeto

Este projeto aborda um problema de negócio crucial para qualquer e-commerce: a capacidade de prever proativamente a insatisfação do cliente. Utilizando o dataset público de E-commerce da Olist, foi desenvolvido um pipeline completo que abrange desde a limpeza e tratamento dos dados brutos até o treino, avaliação e implantação de um modelo de classificação.

O objetivo final não é apenas criar um modelo preciso, mas também disponibilizá-lo como uma ferramenta prática e acessível através de uma API RESTful e uma interface de utilizador simples e intuitiva.

✨ Funcionalidades

    Pipeline de Dados Automatizado: Um script (Pipeline_dados.py) que extrai, limpa, transforma e prepara os dados para modelagem.

    Experimentação e Seleção de Modelos: Um segundo pipeline (Pipeline_modelos.py) que treina múltiplos modelos candidatos (Regressão Logística, Random Forest, LightGBM, XGBoost), seleciona o "campeão" com base na performance e o salva como um ficheiro binário (.joblib).

    Reenquadramento do Problema: Transformação do problema de classificação de 5 classes (notas 1-5) para um problema binário (Satisfeito vs. Insatisfeito), resultando num modelo mais robusto e com maior valor de negócio.

    API RESTful com FastAPI: Um serviço de API (servico_api.py) que carrega o modelo treinado e expõe um endpoint /predict para realizar previsões.

    Interface Web Interativa: Uma página index.html moderna e responsiva que serve como frontend para a API, permitindo a interação com o modelo de forma visual e amigável, com menus de seleção (dropdowns) preenchidos dinamicamente.

    Execução Simplificada: A API foi configurada para servir a interface web e abrir o navegador automaticamente na inicialização, proporcionando uma experiência de utilizador fluida.

🛠️ Tecnologias Utilizadas

    Linguagem: Python 3

    Manipulação de Dados: Pandas, KaggleHub

    Machine Learning: Scikit-learn, XGBoost, LightGBM

    API e Servidor Web: FastAPI, Uvicorn

    Frontend: HTML5, Tailwind CSS, JavaScript

    Serialização de Modelos: Joblib

    Visualização de Dados: Matplotlib, Seaborn

🚀 Configuração e Instalação

Siga os passos abaixo para configurar e executar o projeto no seu ambiente local.
Pré-requisitos

    Python 3.8 ou superior

    pip e venv (geralmente incluídos com o Python)

Passos

    Clone o Repositório (se estiver no Git)

    git clone https://github.com/seu-usuario/seu-repositorio.git
    cd seu-repositorio

    Se não estiver a usar Git, apenas certifique-se de que todos os ficheiros do projeto estão na mesma pasta.

    Crie e Ative um Ambiente Virtual

    # Criar o ambiente virtual
    python -m venv venv

    # Ativar no Windows (PowerShell)
    .\venv\Scripts\Activate.ps1

    # Ativar no macOS/Linux
    source venv/bin/activate

    Instale as Dependências
    Com o ambiente virtual ativo, instale todas as bibliotecas necessárias a partir do ficheiro requirements.txt.

    pip install -r requirements.txt

▶️ Como Executar o Projeto

A execução deve seguir uma ordem específica, pois cada passo depende do anterior.
1. Executar o Pipeline de Dados

Este passo irá descarregar os dados do Kaggle, processá-los e criar o ficheiro output/dados_processados.csv.

python Pipeline_dados.py

2. Executar o Pipeline de Modelagem

Este passo irá carregar os dados processados, treinar os modelos, selecionar o campeão e criar o ficheiro output/modelo_campeao.joblib.

python Pipeline_modelos.py

3. Iniciar o Serviço da API e a Interface

Este é o passo final. Ele irá iniciar o servidor web, que carregará o modelo e abrirá a interface no seu navegador automaticamente.

uvicorn servico_api:app --reload

Após executar o comando, o seu navegador abrirá em http://127.0.0.1:8000, e poderá começar a fazer previsões!


👨‍💻 Autores

Victor Gabriel e Adriane Kelle

Este projeto foi desenvolvido como uma demonstração completa de um ciclo de vida de um projeto de Machine Learning, desde a conceção até à implantação.

