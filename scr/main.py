import streamlit as st
import pandas as pd
import plotly.express as px
from statsbombpy.sb import events, matches, competitions
from typing import List, Dict
import datetime
import os
import google.generativeai as genai
from transformers import pipeline
import numpy as np

HUGGINGFACE_TOKEN = "retirada"  
GEMINI_TOKEN = "retirada"       

st.set_page_config(
    page_title="Análise de Futebol",
    page_icon="⚽",
    layout="wide"
)

@st.cache_resource
def configurar_modelo_linguagem():
    """Configura o modelo de linguagem (Gemini ou HuggingFace)"""
    try:
        genai.configure(api_key=GEMINI_TOKEN)
        modelo = genai.GenerativeModel("gemini-pro")
        return {"tipo": "gemini", "modelo": modelo}
    except Exception as gemini_error:
        try:
            modelo = pipeline(
                "text-generation",
                model="gpt2",
                max_length=500,
                num_return_sequences=1,
                temperature=0.8
            )
            return {"tipo": "huggingface", "modelo": modelo}
        except Exception as huggingface_error:
            st.error(f"Erro ao configurar modelos: Gemini ({str(gemini_error)}), HuggingFace ({str(huggingface_error)})")
            return None

@st.cache_data
def carregar_competicoes():
    """Carrega todas as competições disponíveis"""
    return competitions()

@st.cache_data
def carregar_partidas(id_competicao: int, id_temporada: int):
    """Carrega partidas para competição e temporada específicas"""
    return matches(competition_id=id_competicao, season_id=id_temporada)

def obter_competicoes() -> List[Dict]:
    """Retorna lista de competições disponíveis"""
    df_comp = carregar_competicoes()
    return df_comp.to_dict("records")

def obter_temporadas(id_competicao: int) -> List[int]:
    """Retorna temporadas disponíveis para uma competição"""
    df_comp = carregar_competicoes()
    temporadas = df_comp[df_comp["competition_id"] == id_competicao]["season_id"].unique()
    return sorted(temporadas)

def obter_times_por_competicao(id_competicao: int, id_temporada: int) -> List[str]:
    """Retorna times de uma competição específica"""
    df_partidas = carregar_partidas(id_competicao, id_temporada)
    times = sorted(set(df_partidas["home_team"].unique()) | 
                  set(df_partidas["away_team"].unique()))
    return times

def obter_datas_disponiveis(id_competicao: int, id_temporada: int, time_casa: str, time_fora: str) -> List[datetime.date]:
    """Retorna datas disponíveis para os times selecionados"""
    df_partidas = carregar_partidas(id_competicao, id_temporada)
    partidas_filtradas = df_partidas[
        (df_partidas["home_team"] == time_casa) &
        (df_partidas["away_team"] == time_fora)
    ]
    return sorted(pd.to_datetime(partidas_filtradas["match_date"]).dt.date.unique())

@st.cache_data
def carregar_partida_especifica(id_competicao: int, id_temporada: int, time_casa: str, time_fora: str, data: datetime.date) -> pd.DataFrame:
    """Carrega dados de uma partida específica"""
    df_partidas = carregar_partidas(id_competicao, id_temporada)
    partida = df_partidas[
        (df_partidas["home_team"] == time_casa) &
        (df_partidas["away_team"] == time_fora) &
        (pd.to_datetime(df_partidas["match_date"]).dt.date == data)
    ]
    
    if not partida.empty:
        dados_partida = events(match_id=partida.iloc[0]["match_id"])
        if "player" in dados_partida.columns:
            dados_partida["player"] = dados_partida["player"].fillna("Desconhecido").astype(str)
        return dados_partida
    return None

def calcular_estatisticas_jogador(dados_partida: pd.DataFrame, nome_jogador: str) -> Dict:
    """Calcula estatísticas detalhadas do jogador"""
    if dados_partida is None:
        return None
    
    dados_jogador = dados_partida[dados_partida["player"] == nome_jogador]
    
    estatisticas = {
        "Passes": len(dados_jogador[dados_jogador["type"] == "Pass"]),
        "Passes Completos": len(dados_jogador[
            (dados_jogador["type"] == "Pass") & 
            (dados_jogador["pass_outcome"].isna())
        ]),
        "Finalizações": len(dados_jogador[dados_jogador["type"] == "Shot"]),
        "Gols": len(dados_jogador[
            (dados_jogador["type"] == "Shot") & 
            (dados_jogador["shot_outcome"] == "Goal")
        ]),
        "Desarmes": len(dados_jogador[dados_jogador["type"] == "Tackle"]),
        "Interceptações": len(dados_jogador[dados_jogador["type"] == "Interception"]),
        "Dribles": len(dados_jogador[dados_jogador["type"] == "Dribble"]),
        "Duelos Aéreos": len(dados_jogador[dados_jogador["type"] == "Aerial"]),
        "Faltas Cometidas": len(dados_jogador[dados_jogador["type"] == "Foul Committed"]),
        "Faltas Sofridas": len(dados_jogador[dados_jogador["type"] == "Foul Won"])
    }
    
    if estatisticas["Passes"] > 0:
        estatisticas["Precisão de Passes"] = round(
            (estatisticas["Passes Completos"] / estatisticas["Passes"]) * 100, 1
        )
    else:
        estatisticas["Precisão de Passes"] = 0
        
    return estatisticas

def gerar_resumo_partida(dados_partida: pd.DataFrame) -> str:
    """Gera um resumo detalhado da partida"""
    if dados_partida is None:
        return "Dados da partida não disponíveis"
    
    eventos_importantes = []
    
    gols = dados_partida[
        (dados_partida["type"] == "Shot") & 
        (dados_partida["shot_outcome"] == "Goal")
    ]
    
    for _, gol in gols.iterrows():
        eventos_importantes.append({
            "minuto": gol["minute"],
            "tipo": "Gol",
            "jogador": gol["player"],
            "time": gol["team"]
        })
    
    cartoes = dados_partida[dados_partida["type"] == "Card"]
    for _, cartao in cartoes.iterrows():
        eventos_importantes.append({
            "minuto": cartao["minute"],
            "tipo": f"Cartão {cartao['card_type']}",
            "jogador": cartao["player"],
            "time": cartao["team"]
        })
    
    eventos_importantes.sort(key=lambda x: x["minuto"])
    
    if not eventos_importantes:
        return "Nenhum evento importante registrado na partida"
    
    resumo = "Eventos importantes da partida:\n\n"
    for evento in eventos_importantes:
        resumo += f"{evento['minuto']}' - {evento['tipo']}: {evento['jogador']} ({evento['time']})\n"
    
    return resumo

def gerar_narracao(dados_partida: pd.DataFrame, estilo: str) -> str:
    """Gera narração personalizada da partida usando modelo de linguagem"""
    config_modelo = configurar_modelo_linguagem()
    if config_modelo is None:
        return "Não foi possível gerar a narração. Configure um token válido."
    
    resumo = gerar_resumo_partida(dados_partida)
    
    estilos = {
        "Formal": "de forma técnica e objetiva, como um comentarista profissional",
        "Humorístico": "de forma bem-humorada e descontraída, com analogias engraçadas",
        "Técnico": "com foco em análise tática e técnica, detalhando as jogadas"
    }
    
    prompt = f"""
    Gere uma narração {estilos[estilo]} para esta partida de futebol:
    
    {resumo}
    
    Limite a narração a 3 parágrafos.
    """
    
    try:
        if config_modelo["tipo"] == "gemini":
            resposta = config_modelo["modelo"].generate_content(prompt)
            return resposta.text
        else:  
            resposta = config_modelo["modelo"](prompt)[0]["generated_text"]
            return resposta
    except Exception as e:
        return f"Erro ao gerar narração: {str(e)}"
def main():
    st.title("⚽ Análise de Partidas de Futebol")
    
    with st.container():
        st.subheader("Selecione a Partida")
        
        col1, col2 = st.columns(2)
        
        competicoes = obter_competicoes()
        nomes_comp = [f"{c['competition_name']} ({c['competition_id']})" for c in competicoes]
        
        with col1:
            comp_selecionada = st.selectbox(
                "Competição",
                nomes_comp
            )
            id_competicao = int(comp_selecionada.split("(")[-1].strip(")"))
        
        temporadas = obter_temporadas(id_competicao)
        
        with col2:
            id_temporada = st.selectbox("Temporada", temporadas)
        
        times = obter_times_por_competicao(id_competicao, id_temporada)
        
        col3, col4 = st.columns(2)
        
        with col3:
            time_casa = st.selectbox("Time Casa", times)
        
        with col4:
            times_fora = [t for t in times if t != time_casa]
            time_fora = st.selectbox("Time Visitante", times_fora)
        
        datas_disponiveis = obter_datas_disponiveis(
            id_competicao, id_temporada, time_casa, time_fora
        )
        
        if datas_disponiveis:
            data_partida = st.selectbox(
                "Data da Partida",
                datas_disponiveis,
                format_func=lambda x: x.strftime("%d/%m/%Y")
            )
        else:
            st.warning("Nenhuma partida encontrada para estes times")
            st.stop()
    
    dados_partida = carregar_partida_especifica(
        id_competicao, id_temporada, time_casa, time_fora, data_partida
    )
    
    if dados_partida is None:
        st.error("Erro ao carregar dados da partida")
        st.stop()
    
    aba1, aba2, aba3, aba4 = st.tabs([
        "Análise Individual", 
        "Comparação de Jogadores", 
        "Linha do Tempo",
        "Resumo da Partida"
    ])
    
    with aba1:
        st.subheader("Análise Individual do Jogador")
        
        jogadores = sorted(dados_partida["player"].unique())
        jogador_selecionado = st.selectbox("Selecione o Jogador", jogadores)
        
        if jogador_selecionado:
            estatisticas = calcular_estatisticas_jogador(dados_partida, jogador_selecionado)
            
            if estatisticas:
                colunas = st.columns(len(estatisticas))
                for coluna, (metrica, valor) in zip(colunas, estatisticas.items()):
                    coluna.metric(metrica, valor)
                
                fig = px.bar(
                    x=list(estatisticas.keys()),
                    y=list(estatisticas.values()),
                    title=f"Estatísticas de {jogador_selecionado}",
                    labels={"x": "Métricas", "y": "Quantidade"}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with aba2:
        st.subheader("Comparação entre Jogadores")
        
        col1, col2 = st.columns(2)
        
        with col1:
            jogador1 = st.selectbox("Primeiro Jogador", jogadores, key="j1")
        with col2:
            jogador2 = st.selectbox("Segundo Jogador", jogadores, key="j2")
        
        if jogador1 and jogador2:
            estat1 = calcular_estatisticas_jogador(dados_partida, jogador1)
            estat2 = calcular_estatisticas_jogador(dados_partida, jogador2)
            
            if estat1 and estat2:
                df_comp = pd.DataFrame({
                    jogador1: estat1,
                    jogador2: estat2
                })
                
                fig = px.bar(
                    df_comp,
                    barmode="group",
                    title="Comparação de Estatísticas",
                    labels={"value": "Quantidade", "variable": "Jogador"}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with aba3:
        st.subheader("Linha do Tempo da Partida")
        
        tipos_eventos = sorted(dados_partida["type"].unique())
        eventos_padrao = ["Shot"] if "Shot" in tipos_eventos else []
        
        eventos_selecionados = st.multiselect(
            "Tipos de Eventos",
            tipos_eventos,
            default=eventos_padrao
        )
        
        intervalo_tempo = st.slider("Minutos", 0, 90, (0, 90))
        
        eventos_filtrados = dados_partida[
            (dados_partida["type"].isin(eventos_selecionados)) &
            (dados_partida["minute"].between(intervalo_tempo[0], intervalo_tempo[1]))
        ].sort_values("minute")
        
        for _, evento in eventos_filtrados.iterrows():
            st.markdown(
                f"""
                **{evento['minute']}'** - {evento['player']} - {evento['type']}
                {f"({evento['shot_outcome']})" if evento['type'] == "Shot" else ""}
                """
            )
    
    with aba4:
        st.subheader("Resumo e Narração da Partida")
        
        st.markdown("### Resumo da Partida")
        resumo = gerar_resumo_partida(dados_partida)
        st.text(resumo)
        
        st.markdown("### Narração da Partida")
        estilo_narracao = st.selectbox(
            "Estilo de Narração",
            ["Formal", "Humorístico", "Técnico"]
        )
        
        if st.button("Gerar Narração"):
            with st.spinner("Gerando narração..."):
                narracao = gerar_narracao(dados_partida, estilo_narracao)
                st.markdown(narracao)

if __name__ == "__main__":
    main()    
