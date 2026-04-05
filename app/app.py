
import streamlit as st
import sys
import os
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.predictor import predizer

# ── Configuração da página ─────────────────────────────────
st.set_page_config(
    page_title="Credit Scoring — Give Me Some Credit",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Credit Scoring — Análise de Risco de Crédito")
st.markdown("Preencha os dados do cliente para obter a análise de risco de inadimplência.")
st.divider()

# ── Sidebar — Formulário ───────────────────────────────────
with st.sidebar:
    st.header("📋 Dados do Cliente")

    age = st.slider("Idade", min_value=18, max_value=109, value=35)

    monthly_income = st.number_input(
        "Renda Mensal (R$) — deixe 0 se não souber",
        min_value=0, max_value=100000, value=5400, step=100
    )

    revolving = st.slider(
        "Utilização do Crédito Rotativo (0 a 1)",
        min_value=0.0, max_value=1.1, value=0.3, step=0.01
    )

    debt_ratio = st.slider(
        "Debt Ratio (dívida/renda)",
        min_value=0.0, max_value=1.4, value=0.3, step=0.01
    )

    dependents = st.number_input(
        "Número de Dependentes — deixe -1 se não souber",
        min_value=-1, max_value=20, value=0
    )

    st.subheader("📅 Histórico de Atrasos")
    atraso_30 = st.number_input("Atrasos 30-59 dias", min_value=0, max_value=20, value=0)
    atraso_60 = st.number_input("Atrasos 60-89 dias", min_value=0, max_value=20, value=0)
    atraso_90 = st.number_input("Atrasos 90+ dias",   min_value=0, max_value=20, value=0)

    st.subheader("💳 Linhas de Crédito")
    open_credit = st.number_input("Linhas de crédito abertas", min_value=0, max_value=50, value=5)
    real_estate = st.number_input("Empréstimos imobiliários",  min_value=0, max_value=20, value=1)

    analisar = st.button("🔍 Analisar Cliente", use_container_width=True)

# ── Painel principal ───────────────────────────────────────
if analisar:
    # Preparar dados brutos
    dados = {
        "RevolvingUtilizationOfUnsecuredLines": revolving,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": atraso_30,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": None if monthly_income == 0 else float(monthly_income),
        "NumberOfOpenCreditLinesAndLoans": open_credit,
        "NumberOfTimes90DaysLate": atraso_90,
        "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": atraso_60,
        "NumberOfDependents": None if dependents == -1 else float(dependents),
    }

    with st.spinner("Analisando perfil do cliente..."):
        resultado = predizer(dados)

    proba = resultado["probabilidade"]
    decisao = resultado["decisao"]

    # ── Resultado principal ────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Probabilidade de Inadimplência",
            value=f"{proba:.1%}"
        )

    with col2:
        if decisao == "APROVAR":
            st.success(f"✅ DECISÃO: {decisao}")
        else:
            st.error(f"❌ DECISÃO: {decisao}")

    with col3:
        threshold = resultado["threshold"]
        if proba < threshold * 0.4:
            nivel = "🟢 Baixo Risco"
        elif proba < threshold * 0.7:
            nivel = "🟡 Médio Risco"
        elif proba < threshold:
            nivel = "🟠 Alto Risco — Aprovado com Cautela"
        else:
            nivel = "🔴 Risco Crítico — Reprovado"
        st.metric(label="Nível de Risco", value=nivel)
        
    if decisao == "APROVAR" and proba > threshold * 0.7:
        st.warning(
            f"⚠️ **Atenção:** Cliente aprovado mas com probabilidade de "
            f"inadimplência de {proba:.1%} — próximo ao threshold de reprovação "
            f"({threshold:.0%}). Recomenda-se análise manual adicional."
    )

    st.divider()

    # ── Features processadas ───────────────────────────────
    with st.expander("🔎 Ver features processadas pelo modelo"):
        features = resultado["features_processadas"]
        df_features = json.dumps(features, indent=2, ensure_ascii=False)
        st.json(df_features)

    # ── Fatores de risco identificados ────────────────────
    st.subheader("⚠️ Fatores de Risco Identificados")

    fatores = []
    f = resultado["features_processadas"]

    if f.get("teve_qualquer_atraso", 0) == 1:
        fatores.append("📛 Cliente possui histórico de atrasos")
    if f.get("NumberOfTimes90DaysLate", 0) > 0:
        fatores.append(f"🚨 {int(f['NumberOfTimes90DaysLate'])} atraso(s) grave(s) acima de 90 dias")
    if f.get("RevolvingUtilizationOfUnsecuredLines", 0) > 0.7:
        fatores.append(f"💳 Utilização de crédito rotativo alta: {f['RevolvingUtilizationOfUnsecuredLines']:.0%}")
    if f.get("DebtRatio", 0) > 0.5:
        fatores.append(f"📊 Debt ratio elevado: {f['DebtRatio']:.2f}")
    if f.get("flag_missing_income", 0) == 1:
        fatores.append("❓ Renda não informada — perfil de maior incerteza")

    if fatores:
        for fator in fatores:
            st.warning(fator)
    else:
        st.success("✅ Nenhum fator de risco crítico identificado")

    st.divider()
    st.caption(f"Modelo: XGBoost | Threshold: {resultado['threshold']} | AUC: 0.8697 | KS: 0.5800")

else:
    st.info("👈 Preencha os dados do cliente na barra lateral e clique em **Analisar Cliente**.")