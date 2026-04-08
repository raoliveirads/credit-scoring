# Credit Scoring — Previsão de Inadimplência

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)
![Status](https://img.shields.io/badge/Status-Completo-green)

## Contexto de Negócio

Inadimplência é um dos principais riscos operacionais de instituições financeiras.
Este projeto desenvolve um modelo de machine learning para prever a probabilidade de um cliente não honrar seus compromissos financeiros nos próximos 24 meses, apoiando decisões de concessão de crédito com base em dados históricos de comportamento.

**Problema:** Classificação binária. Cliente vai ou não vai inadimplir?

> Simulação de impacto financeiro baseada em parâmetros hipotéticos
> (ticket médio R$15.000, taxa de recuperação 40%). Valores para fins ilustrativos.
> Com os parâmetros assumidos, o modelo bloquearia ~R$9.9M em perdas no conjunto de teste.

---

## Objetivo

Construir um pipeline completo de Data Science (da exploração dos dados ao deploy) capaz de:
- Identificar clientes de alto risco antes da concessão do crédito
- Explicar as decisões do modelo para stakeholders e compliance
- Operar em produção via interface web com pipeline reproduzível

---

## Dataset

- **Fonte:** [Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
- **Volume:** 150.000 clientes, 11 features originais
- **Target:** `SeriousDlqin2yrs` — inadimplência nos últimos 2 anos
- **Desbalanceamento:** 93.3% adimplentes / 6.7% inadimplentes

---

## Arquitetura do Projeto
```
credit-scoring/
├── data/
│   ├── raw/                    # Dados originais — nunca modificados
│   ├── processed/              # Dados limpos e features engineered
│   └── external/
├── notebooks/
│   ├── 01_eda.ipynb            # Exploração, limpeza e análise
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb       # Treinamento e comparação de modelos
│   └── 04_evaluation.ipynb     # Avaliação, SHAP e threshold
├── src/
│   └── features/
│       └── iv.py               # Módulo de Information Value e WoE
├── app/
│   ├── app.py                  # Interface Streamlit
│   └── predictor.py            # Pipeline de inferência
├── models/
│   ├── xgboost_credit.pkl      # Modelo treinado
│   ├── pipeline_params.json    # Parâmetros de pré-processamento
│   └── features.json           # Lista de features na ordem correta
├── reports/                    # Gráficos e análises exportados
├── requirements.txt
└── MLproject
```

---

## Estratégia de Modelagem

### Pré-processamento
- Remoção de 1 registro com `age = 0` (dado inválido)
- **Achado crítico de EDA:** 93% dos valores absurdos de `DebtRatio` correspondiam a clientes sem renda informada — problema sistêmico de cálculo na origem dos dados
- Winsorização de `RevolvingUtilization` (p99), `MonthlyIncome` (p99) e `DebtRatio` (p99 do subconjunto plausível — winsorização contextual)
- Criação de flags de missing antes da imputação — missing confirmado como não-aleatório
- Imputação com medianas calculadas exclusivamente no conjunto de treino

### Feature Engineering
- **Features binárias:** `teve_atraso_90dias`, `teve_qualquer_atraso` — análise de WoE revelou salto de risco 9x com apenas 1 ocorrência de atraso grave
- **Feature de razão:** `renda_per_capita` — IV de 0.096 supera `MonthlyIncome` isolado (IV 0.067) ao incorporar número de dependentes
- **Score de risco:** índice composto ponderado dos 3 sinais mais fortes, normalizado com limites calculados no treino

### Modelos Comparados

| Modelo | ROC-AUC | KS | F1 (threshold 0.5) | CV AUC | CV Std |
|---|---|---|---|---|---|
| Regressão Logística | 0.8605 | 0.5633 | 0.3338 | 0.8535 | 0.0069 |
| Random Forest | 0.8667 | 0.5819 | 0.3437 | 0.8606 | 0.0068 |
| **XGBoost** | **0.8697** | 0.5800 | 0.3423 | **0.8623** | **0.0060** |

**Modelo escolhido:** XGBoost — maior AUC e menor variância na validação cruzada.
Random Forest teve KS marginalmente superior (0.5819 vs 0.5800) — em contextos que priorizam KS como métrica principal, seria uma alternativa válida.

### Threshold
- Threshold padrão (0.5): Recall 77.4%, Precision 22.0%, F1 0.3423
- **Threshold otimizado (0.749):** F1 0.4538, Recall 54.9%, Precision 38.7%
- Otimização por maximização de F1 — ajustável conforme apetite de risco do negócio

---

## Resultados Finais (threshold 0.749)

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.8697 |
| KS | 0.5800 |
| F1 | 0.4538 |
| Recall | 54.9% |
| Precision | 38.7% |
| Verdadeiros Positivos | 1.101 inadimplentes bloqueados |
| Falsos Positivos | 1.746 bons clientes reprovados |
| Falsos Negativos | 904 inadimplentes aprovados |

**Análise de fairness:**
- Variação de AUC entre faixas etárias: 0.022 — sem evidência de discriminação
- Variação de AUC entre faixas de renda: 0.019 — modelo performa consistentemente
  entre clientes de baixa e alta renda

---

## Interpretabilidade

SHAP values calculados para amostra de 2.000 clientes do conjunto de teste.
Features mais impactantes (ordem de importância global SHAP):

1. `teve_qualquer_atraso` — criada no feature engineering
2. `score_risco` — índice composto criado no feature engineering
3. `total_atrasos` — criada no feature engineering
4. `RevolvingUtilizationOfUnsecuredLines`
5. `age`

As 3 features mais importantes foram criadas durante o feature engineering — confirmando o valor agregado pelo processo de transformação de dados.

---

## Limitações

- Dataset com viés temporal desconhecido — período de coleta não documentado
- `DebtRatio` com problema sistêmico de qualidade para ~20% dos registros — tratado via winsorização contextual, mas feature perde confiabilidade para esse grupo
- Modelo não testado em dados fora da distribuição original (out-of-time validation)
- Threshold fixo — em produção deveria ser recalibrado periodicamente com PSI/KS
- SHAP calculado em amostra de 2.000 registros — não cobre todo o conjunto de teste

---

## Como Rodar

### Instalação
```bash
git clone https://github.com/raoliveirads/credit-scoring
cd credit-scoring
pip install -r requirements.txt
```

### Reproduzir o pipeline completo
```bash
# Executar notebooks na ordem
jupyter notebook notebooks/eda.ipynb
jupyter notebook notebooks/feature_engineering.ipynb
jupyter notebook notebooks/modeling.ipynb
jupyter notebook notebooks/evaluation.ipynb
```

### Rodar o app
```bash
streamlit run app/app.py
```

### MLflow — visualizar experimentos registrados
```bash
mlflow ui
# Acesse: http://localhost:5000
# Experimento: credit-scoring
# Runs registradas: XGBoost_final, XGBoost_final_v2 (com signature)
```

---

## Stack

- **Linguagem:** Python 3.10+
- **ML:** scikit-learn, XGBoost
- **Interpretabilidade:** SHAP
- **Experimentos:** MLflow
- **Deploy:** Streamlit
- **Análise:** pandas, numpy, matplotlib, seaborn

---

## Próximos Passos

- [ ] Containerização com Docker e publicação no Docker Hub
- [ ] Monitoramento de data drift com PSI mensal
- [ ] Validação out-of-time — testar modelo em período diferente do treino
- [ ] Retraining automático com novos dados
- [ ] API REST com FastAPI para integração com sistemas externos
- [ ] Teste A/B entre threshold conservador (0.5) e otimizado (0.749)
