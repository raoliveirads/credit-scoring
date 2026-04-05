import pickle
import json
import numpy as np
import pandas as pd
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def carregar_artefatos():
    """Carrega modelo e parâmetros salvos no treino."""
    with open(os.path.join(ROOT, "models", "xgboost_credit.pkl"), "rb") as f:
        modelo = pickle.load(f)
    with open(os.path.join(ROOT, "models", "pipeline_params.json"), "r") as f:
        params = json.load(f)
    return modelo, params


def preprocessar(dados_brutos: dict, params: dict) -> pd.DataFrame:
    """
    Aplica o mesmo pipeline de pré-processamento do treino.
    Recebe dados brutos do formulário e retorna DataFrame pronto para o modelo.

    Parâmetros:
        dados_brutos : dicionário com campos do formulário
        params       : parâmetros calculados no treino (medianas, limites)

    Retorna:
        DataFrame com 15 features na ordem correta
    """
    d = dados_brutos.copy()

    # ── 1. Flags de missing ────────────────────────────────
    d["flag_missing_income"] = 1 if d["MonthlyIncome"] is None else 0
    d["flag_missing_dependents"] = 1 if d["NumberOfDependents"] is None else 0

    # ── 2. Imputação com medianas do treino ────────────────
    if d["MonthlyIncome"] is None:
        d["MonthlyIncome"] = params["mediana_monthly_income"]
    if d["NumberOfDependents"] is None:
        d["NumberOfDependents"] = params["mediana_number_of_dependents"]

    # ── 3. Winsorização com limites do treino ──────────────
    d["RevolvingUtilizationOfUnsecuredLines"] = min(
        d["RevolvingUtilizationOfUnsecuredLines"],
        params["p99_revolving"]
    )
    d["MonthlyIncome"] = min(d["MonthlyIncome"], params["p99_monthly_income"])
    d["DebtRatio"] = min(d["DebtRatio"], params["p99_debt_ratio"])

    # ── 4. Features derivadas ──────────────────────────────
    d["renda_per_capita"] = d["MonthlyIncome"] / (d["NumberOfDependents"] + 1)

    d["teve_atraso_90dias"] = int(d["NumberOfTimes90DaysLate"] > 0)
    d["teve_qualquer_atraso"] = int(
        d["NumberOfTimes90DaysLate"] > 0 or
        d["NumberOfTime60-89DaysPastDueNotWorse"] > 0 or
        d["NumberOfTime30-59DaysPastDueNotWorse"] > 0
    )
    d["total_atrasos"] = (
        d["NumberOfTime30-59DaysPastDueNotWorse"] +
        d["NumberOfTime60-89DaysPastDueNotWorse"] +
        d["NumberOfTimes90DaysLate"]
    )

    # ── 5. Score de risco (normalizado com limites do treino) ──
    def normalizar(valor, minv, maxv):
        return (valor - minv) / (maxv - minv + 1e-8)

    d["score_risco"] = (
        0.5 * normalizar(
            d["RevolvingUtilizationOfUnsecuredLines"],
            params["revolving_min"], params["revolving_max"]
        ) +
        0.3 * normalizar(
            d["NumberOfTimes90DaysLate"],
            params["atrasos_min"], params["atrasos_max"]
        ) +
        0.2 * normalizar(
            d["total_atrasos"],
            params["atrasos_min"], params["atrasos_max"]
        )
    )

    # ── 6. Montar DataFrame na ordem correta ──────────────
    df = pd.DataFrame([d])[params["features_ordem"]]
    return df


def predizer(dados_brutos: dict) -> dict:
    """
    Pipeline completo de inferência.
    Retorna probabilidade, decisão e threshold usado.
    """
    modelo, params = carregar_artefatos()
    df = preprocessar(dados_brutos, params)

    proba = modelo.predict_proba(df)[0][1]
    threshold = params["threshold_producao"]
    decisao = "REPROVAR" if proba >= threshold else "APROVAR"

    return {
        "probabilidade": round(float(proba), 4),
        "decisao": decisao,
        "threshold": threshold,
        "features_processadas": df.to_dict(orient="records")[0]
    }