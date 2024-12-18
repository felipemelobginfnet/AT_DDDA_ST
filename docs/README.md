# 📊 Análise de Partidas de Futebol

## 📝 Descrição
Uma aplicação web desenvolvida com Streamlit para análise detalhada de partidas de futebol utilizando dados do StatsBomb. A ferramenta permite visualizar estatísticas individuais de jogadores, comparar desempenhos, acompanhar eventos da partida em uma linha do tempo e gerar narrações personalizadas utilizando IA.

## ✨ Funcionalidades

- **Análise Individual**: Visualização detalhada das estatísticas de cada jogador
- **Comparação entre Jogadores**: Análise comparativa de desempenho entre dois atletas
- **Linha do Tempo**: Acompanhamento dos eventos da partida minuto a minuto
- **Resumo e Narração**: Geração automática de resumos e narrações personalizadas usando IA

## 🚀 Configuração do Ambiente

### Pré-requisitos
- Python 3.9 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITORIO]
cd analise-futebol
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
```bash
export HUGGINGFACE_TOKEN="seu_token_aqui"    # Linux/Mac
set HUGGINGFACE_TOKEN="seu_token_aqui"       # Windows

export GEMINI_TOKEN="seu_token_aqui"         # Linux/Mac
set GEMINI_TOKEN="seu_token_aqui"            # Windows
```

### Executando a Aplicação

1. Inicie o servidor Streamlit:
```bash
streamlit run main.py
```

2. Acesse a aplicação em seu navegador através do endereço: `http://localhost:8501`

## 📌 Exemplo de Uso

1. **Seleção da Partida**:
   - Escolha a competição
   - Selecione a temporada
   - Defina os times e a data da partida

2. **Análise Individual**:
   - Selecione um jogador
   - Visualize estatísticas como:
     - Passes completados
     - Finalizações
     - Gols
     - Desarmes
     - Interceptações
     - Dribles
     - Duelos aéreos
     - Faltas

3. **Comparação de Jogadores**:
   - Selecione dois jogadores
   - Compare suas estatísticas em gráficos interativos

4. **Linha do Tempo**:
   - Filtre eventos específicos
   - Defina intervalo de tempo
   - Acompanhe a sequência de acontecimentos

5. **Resumo e Narração**:
   - Visualize resumo automático da partida
   - Gere narrações em diferentes estilos:
     - Formal
     - Humorístico
     - Técnico

## 🔧 Dependências Principais

- streamlit
- pandas
- plotly
- statsbombpy
- google-generativeai
- transformers

## 📋 Estrutura de Dados de Entrada

Os dados devem estar no formato StatsBomb, contendo as seguintes informações:
```python
{
    "competition_id": int,
    "season_id": int,
    "home_team": str,
    "away_team": str,
    "match_date": datetime,
    "events": [
        {
            "type": str,
            "minute": int,
            "player": str,
            "team": str,
            "shot_outcome": str,
            "pass_outcome": str,
            ...
        }
    ]
}
```

