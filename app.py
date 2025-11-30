import streamlit as st
import pandas as pd
import joblib

# --- Configuração da Página ---
st.set_page_config(page_title="Simulador de Safra SC", page_icon="🌱", layout="centered")

# --- Cabeçalho ---
st.title("🌱 Simulador de Produção Agrícola")
st.markdown("""
Este aplicativo utiliza **Inteligência Artificial (Random Forest)** para estimar a produtividade 
de sementes no estado de **Santa Catarina**, baseado no histórico do SIGEF (2013-2026).
""")
st.divider()

# --- 1. Carregar o Cérebro da IA ---
@st.cache_resource
def carregar_ia():
    try:
        modelo = joblib.load('modelo_sementes_rf.joblib')
        le_especie = joblib.load('encoder_especie.joblib')
        le_categoria = joblib.load('encoder_categoria.joblib')
        return modelo, le_especie, le_categoria
    except:
        return None, None, None

modelo, le_especie, le_categoria = carregar_ia()

# --- 2. Interface do Usuário ---
if modelo is None:
    st.error("⚠️ ERRO: Arquivos do modelo não encontrados! Verifique se os arquivos .joblib estão na pasta.")
else:
    # Colunas para organizar a tela
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Dados do Campo")
        area = st.number_input("Área Plantada (hectares)", value=20.0, step=1.0, min_value=0.1)
        ano = st.number_input("Ano da Safra", value=2025, step=1)

    with col2:
        st.subheader("🌽 Cultura")
        # Pega a lista de plantas que a IA aprendeu
        opcoes_especies = sorted(le_especie.classes_)
        
        # Tenta deixar "Soja" (Glycine max) selecionado por padrão se existir
        index_padrao = 0
        if 'Glycine max (L.) Merr.' in opcoes_especies:
            index_padrao = opcoes_especies.index('Glycine max (L.) Merr.')
            
        especie = st.selectbox("Selecione a Espécie", options=opcoes_especies, index=index_padrao)
        categoria = st.selectbox("Categoria da Semente", options=sorted(le_categoria.classes_))

    # --- 3. Botão de Ação ---
    st.divider()
    botao = st.button("🔮 Calcular Previsão", type="primary", use_container_width=True)

    if botao:
        # Preparar os dados para a IA (exatamente como foi no treino)
        cod_esp = le_especie.transform([especie])[0]
        cod_cat = le_categoria.transform([categoria])[0]
        
        # Cria a tabela de 1 linha (seu cenário)
        dados = pd.DataFrame([[area, cod_esp, cod_cat, ano]], 
                             columns=['Area', 'Especie_Encoded', 'Categoria_Encoded', 'Ano_Safra'])
        
        # A Mágica Acontece (Previsão)
        producao_estimada = modelo.predict(dados)[0]
        produtividade = producao_estimada / area

        # --- 4. Exibir Resultados ---
        st.success("Cálculo realizado com sucesso!")
        
        # Cartões de Resultado (Métricas)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Produção Total Esperada", f"{producao_estimada:,.2f} kg")
        with c2:
            st.metric("Produtividade Média", f"{produtividade:,.2f} kg/ha")

        # Feedback visual
        if produtividade > 3000:
            st.balloons() # Solta balões se a produção for alta!