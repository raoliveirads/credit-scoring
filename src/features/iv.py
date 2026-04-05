import numpy as np
import pandas as pd


def calcular_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 10) -> float:
    """
    Calcula o Information Value (IV) de uma feature contínua ou categórica.

    Parâmetros:
        df      : DataFrame com os dados
        feature : Nome da coluna preditora
        target  : Nome da coluna target (binária: 0/1)
        bins    : Número de bins para discretização (default: 10)

    Retorna:
        IV (float)
    """
    df_temp = df[[feature, target]].copy()

    if df_temp[feature].nunique() > 10:
        df_temp["bin"] = pd.qcut(df_temp[feature], q=bins, duplicates="drop")
    else:
        df_temp["bin"] = df_temp[feature]

    total_events = df_temp[target].sum()
    total_non_events = len(df_temp) - total_events

    iv_table = df_temp.groupby("bin", observed=True)[target].agg(
        events="sum",
        total="count"
    )
    iv_table["non_events"] = iv_table["total"] - iv_table["events"]
    iv_table["pct_events"] = iv_table["events"] / total_events
    iv_table["pct_non_events"] = iv_table["non_events"] / total_non_events

    iv_table = iv_table[
        (iv_table["pct_events"] > 0) & (iv_table["pct_non_events"] > 0)
    ]

    iv_table["woe"] = np.log(
        iv_table["pct_events"] / iv_table["pct_non_events"]
    )
    iv_table["iv"] = (
        (iv_table["pct_events"] - iv_table["pct_non_events"]) * iv_table["woe"]
    )

    return iv_table["iv"].sum()


def iv_contagem(df: pd.DataFrame, col: str, target: str) -> float:
    """
    Calcula IV para variáveis de contagem com alta concentração em zero.
    Usa bins de negócio fixos: 0, 1, 2, 3+

    Parâmetros:
        df     : DataFrame com os dados
        col    : Nome da coluna preditora
        target : Nome da coluna target (binária: 0/1)

    Retorna:
        IV (float)
    """
    df_temp = df[[col, target]].copy()

    df_temp["bin"] = pd.cut(
        df_temp[col],
        bins=[-1, 0, 1, 2, 999],
        labels=["0", "1", "2", "3+"]
    )

    total_events = df_temp[target].sum()
    total_non_events = len(df_temp) - total_events

    iv_table = df_temp.groupby("bin", observed=True)[target].agg(
        events="sum", total="count"
    )
    iv_table["non_events"] = iv_table["total"] - iv_table["events"]
    iv_table["pct_events"] = iv_table["events"] / total_events
    iv_table["pct_non_events"] = iv_table["non_events"] / total_non_events

    iv_table = iv_table[
        (iv_table["pct_events"] > 0) & (iv_table["pct_non_events"] > 0)
    ]

    iv_table["woe"] = np.log(
        iv_table["pct_events"] / iv_table["pct_non_events"]
    )
    iv_table["iv"] = (
        (iv_table["pct_events"] - iv_table["pct_non_events"]) * iv_table["woe"]
    )

    return iv_table["iv"].sum()


def classificar_iv(iv: float) -> str:
    """
    Classifica o poder preditivo de uma feature com base no IV.

    Referência padrão de mercado:
        < 0.02  → Fraco
        0.02–0.1 → Médio
        0.1–0.3  → Forte
        > 0.3   → Suspeito (investigar possível leakage)
    """
    if iv < 0.02:
        return "Fraco"
    elif iv < 0.1:
        return "Médio"
    elif iv < 0.3:
        return "Forte"
    else:
        return "Suspeito (investigar leakage)"