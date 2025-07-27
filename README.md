🤖 Preditor de Satisfação de Clientes de E-commerce

Um projeto completo que constrói e implanta um modelo de Machine Learning para prever a satisfação de clientes, servido através de uma API interativa com interface web.

Status do Projeto: Concluído ✅

A interface web permite que o utilizador insira dados de um pedido e receba uma previsão de satisfação em tempo real.

📝 Visão Geral do Projeto

Este projeto aborda um problema de negócio crucial para qualquer e-commerce: a capacidade de prever proativamente a insatisfação do cliente. Utilizando o dataset público de E-commerce da Olist, foi desenvolvido um pipeline completo que abrange desde a limpeza dos dados brutos até o treino e implantação de um modelo de classificação.

O projeto evoluiu para incorporar técnicas avançadas como Processamento de Linguagem Natural (NLP) e reamostragem (SMOTE) para resolver o desafio do desbalanceamento de classes, focando em maximizar a deteção de clientes em risco. O objetivo final é disponibilizar uma ferramenta prática e precisa através de uma API RESTful e uma interface de utilizador intuitiva.

✨ Funcionalidades

    Pipeline de Dados Automatizado: Um script (Pipeline_dados.py) que extrai, limpa, transforma e prepara os dados para modelagem.

    Engenharia de Atributos Avançada com NLP: O pipeline de dados foi aprimorado para incluir e processar os comentários de texto das avaliações, transformando-os em features valiosas para o modelo.

    Tratamento de Desbalanceamento com SMOTE: Implementação da técnica de reamostragem SMOTE para criar um conjunto de treino mais balanceado, melhorando drasticamente a capacidade do modelo de identificar a classe minoritária (clientes insatisfeitos).

    Experimentação e Seleção de Modelos por F1-Score: Um pipeline (Pipeline_modelos.py) que treina múltiplos modelos, mas seleciona o "campeão" com base no F1-Score, uma métrica mais robusta que a acurácia para este tipo de problema.

    API RESTful com FastAPI: Um serviço de API (servico_api.py) que carrega o modelo treinado e expõe endpoints para realizar previsões e popular a interface.

    Interface Web Interativa: Uma página index.html moderna que serve como frontend para a API, com menus de seleção preenchidos dinamicamente.

    Execução Simplificada: A API foi configurada para servir a interface e abrir o navegador automaticamente na inicialização.

🛠️ Tecnologias Utilizadas

    Linguagem: Python 3

    Manipulação de Dados: Pandas, KaggleHub

    Machine Learning: Scikit-learn, XGBoost, LightGBM, Imbalanced-learn (para SMOTE)

    API e Servidor Web: FastAPI, Uvicorn

    Frontend: HTML5, Tailwind CSS, JavaScript

    Serialização de Modelos: Joblib

    Visualização de Dados: Matplotlib, Seaborn

🚀 Configuração e Instalação

Siga os passos abaixo para configurar e executar o projeto no seu ambiente local.
Pré-requisitos:

    Python 3.8 ou superior

    pip e venv (geralmente incluídos com o Python)

    Visual Studio Code

Passos

    Clone o Repositório (se estiver no Git)

    git clone https://github.com/VitNog21/PipelineESI
    cd PipelineESI

    Se não estiver usando Git, apenas certifique-se de que todos os ficheiros do projeto estão na mesma pasta.

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

É crucial que os scripts sejam executados na ordem correta, pois cada passo gera os ficheiros necessários para o próximo.
Opção 1: Usando o Debugger do VS Code (Recomendado)

Esta é a forma mais prática e integrada. Com o ficheiro .vscode/launch.json configurado, pode executar cada etapa com um clique.

    Abra a aba de Execução e Depuração:

        Clique no ícone de "Executar e Depurar" na barra lateral esquerda do VS Code (ou pressione Ctrl+Shift+D).

    Selecione e Execute as Configurações na Ordem Correta:
    No topo da barra lateral, use o menu de seleção para executar cada configuração na seguinte ordem, clicando no botão verde de "play" (▶️):

        1. Executar Pipeline de Dados

        2. Executar Pipeline de Modelagem

        3. Iniciar API e Interface

Opção 2: Usando o Terminal Manualmente

Se preferir não usar o debugger do VS Code, pode executar os comandos diretamente no terminal integrado (com o ambiente virtual ativo).

    Execute o Pipeline de Dados:

    python Pipeline_dados.py

    Execute o Pipeline de Modelagem:

    python Pipeline_modelos.py

    Inicie o Serviço da API e a Interface:

    uvicorn servico_api:app --reload

    (O seu navegador abrirá automaticamente)


👨‍💻 Autores

Victor Gabriel e Adriane Kelle

Este projeto foi desenvolvido como uma demonstração completa de um ciclo de vida de um projeto de Machine Learning, desde a conceção até à implantação e otimização.