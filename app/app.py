import os
import sys

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.predictor import predizer

# Tema claro definido no próprio app (dispensa .streamlit/config.toml)
try:
    for _k, _v in {"theme.base": "light", "theme.primaryColor": "#2c4f86", "theme.backgroundColor": "#f6f5f2",
                   "theme.secondaryBackgroundColor": "#ffffff", "theme.textColor": "#1d2129"}.items():
        st._config.set_option(_k, _v)
except Exception:
    pass

st.set_page_config(page_title="Análise de Crédito", layout="wide")

# ── Configuração ───────────────────────────────────────────
AUC, KS = 0.8697, 0.58

# Rodapé — troque pelos seus dados
AUTOR = "Renan Amaral"
LINKS = [
    ("GitHub", "https://github.com/raoliveirads", "github"),
    ("Portfólio", "https://raoliveirads.github.io/", "globe"),
    ("LinkedIn", "https://www.linkedin.com/in/renanammaral/", "linkedin"),
]

PADRAO = dict(age=35, renda=5400, renda_ni=False, dep=0, dep_ni=False, rev=30, debt=30,
              a30=0, a60=0, a90=0, linhas=5, imob=1)
# Perfil de referência usado para explicar o resultado
REF = {"RevolvingUtilizationOfUnsecuredLines": 0.15, "age": 52, "NumberOfTime30-59DaysPastDueNotWorse": 0,
       "DebtRatio": 0.35, "MonthlyIncome": 6700.0, "NumberOfOpenCreditLinesAndLoans": 8,
       "NumberOfTimes90DaysLate": 0, "NumberRealEstateLoansOrLines": 1,
       "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfDependents": 0.0}

LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Uso do rotativo", "age": "Idade",
    "NumberOfTime30-59DaysPastDueNotWorse": "Atrasos 30–59 dias", "DebtRatio": "Comprometimento da renda",
    "MonthlyIncome": "Renda mensal", "NumberOfOpenCreditLinesAndLoans": "Linhas abertas",
    "NumberOfTimes90DaysLate": "Atrasos 90+ dias", "NumberRealEstateLoansOrLines": "Financiamentos imobiliários",
    "NumberOfTime60-89DaysPastDueNotWorse": "Atrasos 60–89 dias", "NumberOfDependents": "Dependentes",
    "teve_qualquer_atraso": "Teve algum atraso", "flag_missing_income": "Renda ausente",
    "renda_per_capita": "Renda per capita", "teve_atraso_90dias": "Teve atraso 90+ dias",
}

VERDE, AMARELO, LARANJA, VERMELHO = "#3f8f62", "#c9a227", "#d07a3a", "#c2453d"


# ── Utilidades ─────────────────────────────────────────────
def html(s):
    st.markdown("".join(l.strip() for l in s.splitlines()), unsafe_allow_html=True)


def num(v, d=1):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v, d=1):
    return num(v * 100, d) + "%"


def pp(v):
    return ("+" if v >= 0 else "−") + num(abs(v) * 100) + " pp"


def brl(v):
    return "R$ " + num(v, 0)


def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float) and not v.is_integer():
        return num(v, 2)
    return str(int(v))


def prob(d):
    return predizer(d)["probabilidade"]


# ── Estado e callbacks ─────────────────────────────────────
for k, v in PADRAO.items():
    st.session_state.setdefault(k, v)


def restaurar():
    for k, v in PADRAO.items():
        st.session_state[k] = v


def aplicar(mudancas):
    for k, v in mudancas.items():
        st.session_state[k] = v


# ── Estilo ─────────────────────────────────────────────────
html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, .stApp, [class*="css"], button, input { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
.stApp { background: #f6f5f2; color: #1d2129; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1360px; margin: 0 auto; padding: 1.2rem 1.75rem 3rem; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
section[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e3e1db; }
section[data-testid="stSidebar"] hr { margin: 12px 0; border-color: #eeece7; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.card-marker):not(:has([data-testid="stVerticalBlockBorderWrapper"] .card-marker)) { background: #fff; border-color: #e3e1db !important; border-radius: 10px; }
[data-baseweb="input"], [data-baseweb="input"] > div, [data-testid="stNumberInput"] input { background: #fff !important; color: #1d2129 !important; }
[data-testid="stNumberInput"] button { background: #fff !important; color: #4a4e57 !important; }
[data-testid="stNumberInputContainer"] { border: 1px solid #d9d6cf; border-radius: 6px; }
[data-testid="stSlider"] [role="slider"] { background-color: #2c4f86 !important; border-color: #2c4f86 !important; }
[data-testid="stSliderThumbValue"], [data-testid="stThumbValue"] { color: #2c4f86 !important; }
[data-testid="stCheckbox"] input:checked + div, [data-testid="stCheckbox"] label > span:first-child { border-color: #2c4f86 !important; }
[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > span:first-child { background-color: #fff; border: 1px solid #c9c6bf; }
[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked) > span:first-child { background-color: #2c4f86 !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] details { background: #fff; color: #1d2129; }
.stSlider label p, .stNumberInput label p, .stCheckbox label p { font-size: 14px; color: #1d2129; }
.stButton button { border-radius: 6px; border: 1px solid #d9d6cf; background: #fff; color: #1d2129; font-size: 13px; }
.stButton button:hover { background: #f4f2ee; border-color: #c9c6bf; color: #1d2129; }
[data-testid="stExpander"] { background: #fff; border: 1px solid #e3e1db; border-radius: 10px; }
/* formulário compacto (somente o card da esquerda; funciona em versões antigas e novas do Streamlit) */
[data-testid="stVerticalBlock"]:has(.form-marker):not(:has([data-testid="stVerticalBlock"] .form-marker)) { gap: .2rem !important; }
[data-testid="stVerticalBlock"]:has(.form-marker):not(:has([data-testid="stVerticalBlock"] .form-marker)) .eyebrow { margin: 2px 0 0; }
[data-testid="stVerticalBlock"]:has(.form-marker):not(:has([data-testid="stVerticalBlock"] .form-marker)) .hint { margin: -14px 0 0; }
[data-testid="stVerticalBlock"]:has(.form-marker):not(:has([data-testid="stVerticalBlock"] .form-marker)) [data-testid="stSlider"] { margin-bottom: -14px; }
[data-testid="stVerticalBlock"]:has(.form-marker):not(:has([data-testid="stVerticalBlock"] .form-marker)) [data-testid="stSlider"] [data-baseweb="slider"] { padding-top: 4px; }
[data-testid="stVerticalBlock"]:has(.form-marker):not(:has([data-testid="stVerticalBlock"] .form-marker)) [data-testid="stTickBar"] { display: none; }
[data-testid="stVerticalBlock"]:has(.form-marker):not(:has([data-testid="stVerticalBlock"] .form-marker)) [data-testid="stCheckbox"] { margin: 0 0 -6px; }
[data-testid="stVerticalBlock"]:has(.form-marker):not(:has([data-testid="stVerticalBlock"] .form-marker)) [data-testid="stNumberInput"] input { height: 34px; }
[data-testid="stVerticalBlock"]:has(.form-marker):not(:has([data-testid="stVerticalBlock"] .form-marker)) [data-testid="stNumberInputContainer"] { height: 36px; }
[data-testid="stVerticalBlock"]:has(.form-marker):not(:has([data-testid="stVerticalBlock"] .form-marker)) [data-testid="stWidgetLabel"] { margin-bottom: 0; min-height: 1.2rem; }
[data-testid="stVerticalBlock"]:has(.form-marker):not(:has([data-testid="stVerticalBlock"] .form-marker)) [data-testid="stWidgetLabel"] p { font-size: 13px !important; }
[data-testid="stVerticalBlock"]:has(.form-marker):not(:has([data-testid="stVerticalBlock"] .form-marker)) .stElementContainer:has(.card-marker), [data-testid="stVerticalBlock"]:has(.form-marker):not(:has([data-testid="stVerticalBlock"] .form-marker)) .stElementContainer:has(.form-marker),
[data-testid="stVerticalBlock"]:has(.form-marker):not(:has([data-testid="stVerticalBlock"] .form-marker)) [data-testid="stElementContainer"]:has(.card-marker), [data-testid="stVerticalBlock"]:has(.form-marker):not(:has([data-testid="stVerticalBlock"] .form-marker)) [data-testid="stElementContainer"]:has(.form-marker) { display: none; }
hr.sep { border: none; border-top: 1px solid #eeece7; margin: 8px 0 2px; }
.mono { font-family: 'IBM Plex Mono', monospace; }
.eyebrow { font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: #8a8d94; margin: 4px 0 8px; }
.hint { font-size: 12px; color: #8a8d94; margin: -12px 0 10px; }
.h3 { font-size: 15px; font-weight: 600; margin: 0; }
.sub { font-size: 13px; color: #6b6f78; margin-top: 2px; }
.hero { background: #15181e; color: #f3f2ee; border-radius: 12px; overflow: hidden; margin-bottom: 16px; }
.hero-top { padding: 30px 32px 24px; }
.hero-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; flex-wrap: wrap; }
.hero-lbl { font-size: 13px; color: #a2a6ae; }
.hero-num { font-size: 68px; font-weight: 500; letter-spacing: -.04em; line-height: 1; margin-top: 8px; }
.badge { display: inline-block; padding: 9px 16px; border-radius: 6px; font-size: 16px; font-weight: 600; color: #fbfaf8; }
.bar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; margin-top: 28px; }
.marker { position: absolute; top: -16px; width: 4px; height: 22px; margin-left: -2px; background: #fff; border-radius: 2px; box-shadow: 0 0 0 3px #15181e; }
.scale { position: relative; height: 16px; font-size: 11px; color: #8a8e97; margin-top: 8px; }
.stats { display: flex; flex-wrap: wrap; gap: 1px; background: #2a2e36; border-top: 1px solid #2a2e36; }
.stat { flex: 1 1 170px; background: #15181e; padding: 16px 32px; }
.stat span { display: block; font-size: 12px; color: #8a8e97; margin-bottom: 4px; }
.stat b { font-size: 15px; font-weight: 500; }
.note { background: #fbf5e6; border: 1px solid #ecdcae; border-radius: 8px; padding: 14px 16px; font-size: 14px; color: #4a3b12; margin-bottom: 16px; }
.axis { display: flex; justify-content: space-between; font-size: 11px; color: #8a8d94; margin: 14px 0 4px; }
.crow { display: grid; grid-template-columns: 1fr 1fr; row-gap: 6px; padding: 9px 0; border-bottom: 1px solid #f1efea; }
.crow .t { grid-column: 1 / -1; display: flex; justify-content: space-between; gap: 12px; font-size: 13px; }
.neg { display: flex; justify-content: flex-end; height: 8px; border-right: 1px solid #c9c6bf; }
.scen-t { font-size: 14px; font-weight: 500; }
.flip { display: inline-block; font-size: 11px; font-weight: 600; padding: 3px 7px; border-radius: 4px; margin-left: 6px; }
.scen-n { display: flex; align-items: baseline; gap: 10px; margin: 10px 0 6px; }
.rodape { display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap; border-top: 1px solid #e3e1db; margin-top: 32px; padding-top: 18px; font-size: 13px; color: #6b6f78; }
.rodape nav { display: flex; gap: 6px; flex-wrap: wrap; }
.rodape a { display: flex; align-items: center; gap: 8px; padding: 7px 12px; border: 1px solid #e3e1db; border-radius: 6px; background: #fff; color: #1d2129 !important; text-decoration: none; }
.rodape a:hover { background: #f4f2ee; }
.rodape img { width: 16px; height: 16px; opacity: .75; }
.frow { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #f4f2ee; font-size: 13px; }
</style>
""")

# ── Layout: cabeçalho, formulário à esquerda, resultado à direita ──
cabecalho = st.container()
col_form, col_main = st.columns([1, 2.4], gap="large")

with col_form, st.container(border=True):
    html('<span class="card-marker"></span>')
    html('<span class="form-marker"></span>')
    c_t, c_b = st.columns([3, 2], vertical_alignment="center")
    c_t.markdown('<p class="h3">Dados do cliente</p>', unsafe_allow_html=True)
    c_b.button("Restaurar padrão", on_click=restaurar, use_container_width=True)

    html('<div class="eyebrow">Perfil</div>')
    st.slider("Idade", 18, 100, key="age", format="%d anos")
    st.checkbox("Renda não informada", key="renda_ni")
    st.number_input("Renda mensal (R$)", 0, 100000, step=100, key="renda", disabled=st.session_state.renda_ni)
    st.checkbox("Dependentes não informado", key="dep_ni")
    st.number_input("Dependentes", 0, 20, key="dep", disabled=st.session_state.dep_ni)

    html('<hr class="sep"><div class="eyebrow">Endividamento</div>')
    st.slider("Uso do crédito rotativo", 0, 110, step=1, key="rev", format="%d%%")
    html('<div class="hint">Saldo usado em cartões e linhas sem garantia sobre o limite total</div>')
    st.slider("Comprometimento da renda", 0, 140, step=1, key="debt", format="%d%%")
    html('<div class="hint">Pagamentos mensais de dívidas sobre a renda bruta</div>')

    html('<hr class="sep"><div class="eyebrow">Atrasos nos últimos 2 anos</div>')
    st.number_input("30 a 59 dias", 0, 20, key="a30")
    st.number_input("60 a 89 dias", 0, 20, key="a60")
    st.number_input("90 dias ou mais", 0, 20, key="a90")

    html('<hr class="sep"><div class="eyebrow">Linhas de crédito</div>')
    st.number_input("Linhas e empréstimos abertos", 0, 50, key="linhas")
    st.number_input("Financiamentos imobiliários", 0, 20, key="imob")

S = st.session_state


def montar(s):
    return {
        "RevolvingUtilizationOfUnsecuredLines": s["rev"] / 100,
        "age": s["age"],
        "NumberOfTime30-59DaysPastDueNotWorse": s["a30"],
        "DebtRatio": s["debt"] / 100,
        "MonthlyIncome": None if s["renda_ni"] else float(s["renda"]),
        "NumberOfOpenCreditLinesAndLoans": s["linhas"],
        "NumberOfTimes90DaysLate": s["a90"],
        "NumberRealEstateLoansOrLines": s["imob"],
        "NumberOfTime60-89DaysPastDueNotWorse": s["a60"],
        "NumberOfDependents": None if s["dep_ni"] else float(s["dep"]),
    }


estado = {k: S[k] for k in PADRAO}
dados = montar(estado)
resultado = predizer(dados)
p = resultado["probabilidade"]
threshold = resultado["threshold"]
aprovado = resultado["decisao"] == "APROVAR"
features = resultado["features_processadas"]

# ── Cabeçalho ──────────────────────────────────────────────
with cabecalho:
  html(f"""
<div style="display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;padding-bottom:16px;margin-bottom:16px;border-bottom:1px solid #e3e1db">
  <div style="display:flex;gap:12px;align-items:center">
    <div style="width:28px;height:28px;border-radius:6px;background:#1d2129;color:#fbfaf8;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:600">CS</div>
    <div><div style="font-size:15px;font-weight:600">Análise de Crédito</div>
    <div style="font-size:12px;color:#6b6f78">Avaliação de risco de inadimplência em 2 anos</div></div>
  </div>
  <div class="mono" style="display:flex;gap:18px;font-size:12px;color:#6b6f78">
    <span>XGBoost</span><span>AUC {num(AUC, 3)}</span><span>KS {num(KS, 3)}</span><span>Corte {pct(threshold, 0)}</span>
  </div>
</div>
""")

with col_main:
    # ── Nível de risco ─────────────────────────────────────────
    if p < threshold * 0.4:
        nivel, cor_nivel = "Risco baixo", "#8fd1a9"
    elif p < threshold * 0.7:
        nivel, cor_nivel = "Risco moderado", "#e3c86a"
    elif p < threshold:
        nivel, cor_nivel = "Risco alto", "#e9a26e"
    else:
        nivel, cor_nivel = "Risco crítico", "#ec8c80"

    # ── Contribuições (variável trocada pelo valor de referência) ──
    contribs = []
    for chave, ref in REF.items():
        if dados[chave] == ref:
            continue
        d = p - prob({**dados, chave: ref})
        if abs(d) >= 0.001:
            contribs.append((chave, d))
    contribs.sort(key=lambda x: -abs(x[1]))
    contribs = contribs[:7]
    maior_alta = next((LABELS[k] for k, d in contribs if d > 0), "Nenhum acima da referência")


    def valor_legivel(chave):
        v = dados[chave]
        if v is None:
            return "não informado"
        if chave in ("RevolvingUtilizationOfUnsecuredLines", "DebtRatio"):
            return pct(v, 0)
        if chave == "MonthlyIncome":
            return brl(v)
        if chave == "age":
            return f"{int(v)} anos"
        return str(int(v))


    # ── Card principal ─────────────────────────────────────────
    escala = min(1.0, max(threshold * 2, 0.6))
    pos = lambda v: min(v, escala) / escala * 100
    faixa = threshold * 0.3 / escala * 100
    badge_bg = "#2f7a52" if aprovado else "#b23a31"

    html(f"""
    <div class="hero">
      <div class="hero-top">
        <div class="hero-row">
          <div><div class="hero-lbl">Probabilidade de inadimplência</div><div class="hero-num">{pct(p)}</div></div>
          <div style="text-align:right">
            <span class="badge" style="background:{badge_bg}">{"Aprovar" if aprovado else "Reprovar"}</span>
            <div style="font-size:13px;font-weight:500;color:{cor_nivel};margin-top:10px">{nivel}</div>
          </div>
        </div>
        <div class="bar">
          <div style="width:{pos(threshold * 0.4)}%;background:#4d9a6c"></div>
          <div style="width:{faixa}%;background:#c9a640"></div>
          <div style="width:{faixa}%;background:#c97a45"></div>
          <div style="flex:1;background:#b84b40"></div>
        </div>
        <div style="position:relative"><div class="marker" style="left:{pos(p)}%"></div></div>
        <div class="scale mono">
          <span style="position:absolute;left:0">0%</span>
          <span style="position:absolute;left:{pos(threshold)}%;transform:translateX(-50%);color:#f3f2ee">corte {pct(threshold, 0)}</span>
          <span style="position:absolute;right:0">{pct(escala, 0)}{"" if escala >= 1 else "+"}</span>
        </div>
      </div>
      <div class="stats">
        <div class="stat"><span>Distância do corte</span><b class="mono">{pp(p - threshold)}</b></div>
        <div class="stat"><span>Maior peso no risco</span><b>{maior_alta}</b></div>
      </div>
    </div>
    """)

    if aprovado and p > threshold * 0.7:
        html(f'<div class="note"><b>Recomendada revisão manual.</b> Cliente aprovado a {num((threshold - p) * 100)} pontos do corte de reprovação.</div>')

    # ── Simulação ──────────────────────────────────────────────
    cenarios = []
    if estado["rev"] > 30:
        cenarios.append(("Reduzir uso do rotativo para 30%", {"rev": 30}))
    if estado["debt"] > 35:
        cenarios.append(("Reduzir comprometimento da renda para 35%", {"debt": 35}))
    if estado["renda_ni"]:
        cenarios.append((f"Informar renda de {brl(6700)}", {"renda_ni": False, "renda": 6700}))
    if estado["dep_ni"]:
        cenarios.append(("Informar número de dependentes", {"dep_ni": False}))
    if len(cenarios) < 3 and estado["rev"] <= 80:
        novo = min(110, estado["rev"] + 40)
        cenarios.append((f"Rotativo subir para {novo}%", {"rev": novo}))


    def p_com_rotativo(r):
        return prob({**dados, "RevolvingUtilizationOfUnsecuredLines": r})


    if aprovado:
        grade = np.arange(estado["rev"] / 100, 1.1001, 0.02)
        r = next((x for x in grade if p_com_rotativo(x) >= threshold), None)
        virada = "Segue aprovado mesmo com 110% de rotativo" if r is None else f"Reprovaria com rotativo a partir de {pct(r, 0)}"
    else:
        grade = np.arange(estado["rev"] / 100, -0.0001, -0.02)
        r = next((x for x in grade if p_com_rotativo(x) < threshold), None)
        virada = "Nenhum nível de rotativo aprova este perfil" if r is None else f"Aprovaria com rotativo até {pct(max(0, r), 0)}"

    # ── Explicação ─────────────────────────────────────────────
    # Empilhados, como no protótipo: explicação e, logo abaixo, simulação
    col_exp = st.container()
    col_sim = st.container()

    with col_exp:
        with st.container(border=True):
            html('<span class="card-marker"></span>')
            linhas = ""
            mx = max([abs(d) for _, d in contribs] + [0.01])
            for chave, d in contribs:
                cor = "#a8322b" if d > 0 else "#2f7a52"
                w = abs(d) / mx * 100
                linhas += f"""
                <div class="crow">
                  <div class="t"><span><b style="font-weight:500">{LABELS[chave]}</b> <span style="color:#8a8d94">{valor_legivel(chave)}</span></span>
                  <span class="mono" style="color:{cor}">{pp(d)}</span></div>
                  <div class="neg"><div style="width:{w if d < 0 else 0}%;background:#5fa57c;border-radius:3px 0 0 3px"></div></div>
                  <div style="display:flex;height:8px"><div style="width:{w if d > 0 else 0}%;background:#c95a4f;border-radius:0 3px 3px 0"></div></div>
                </div>"""
            if not contribs:
                linhas = '<div class="sub" style="padding:12px 0">Perfil igual ao de referência.</div>'
            html(f"""
            <p class="h3">O que explica o resultado</p>
            <div class="sub">Variação em pontos percentuais em relação a um perfil de referência</div>
            <div class="axis mono"><span>reduz o risco</span><span>aumenta o risco</span></div>
            {linhas}
            """)

    with col_sim:
        with st.container(border=True):
            html('<span class="card-marker"></span>')
            h1, h2 = st.columns([3, 2])
            with h1:
                html('<p class="h3">Simulação</p><div class="sub">O que mudaria a análise mantendo o restante do perfil</div>')
            with h2:
                html(f'<div style="display:flex;justify-content:flex-end"><span style="font-size:13px;background:#f4f2ee;border-radius:6px;padding:6px 10px">{virada}</span></div>')

            if cenarios:
                cols = st.columns(len(cenarios), gap="medium")
                for col, (titulo, mud) in zip(cols, cenarios):
                    p2 = prob(montar({**estado, **mud}))
                    flip = ""
                    if (p2 < threshold) != aprovado:
                        bg, fg, txt = (("#f8e7e4", "#8f2f27", "Passa a reprovar") if aprovado
                                       else ("#e6f3ea", "#2a5e40", "Passa a aprovar"))
                        flip = f'<span class="flip" style="background:{bg};color:{fg}">{txt}</span>'
                    cor = "#a8322b" if p2 > p else "#2f7a52"
                    with col:
                        html(f"""
                        <div style="border-top:1px solid #eeece7;padding-top:14px;margin-top:4px">
                          <span class="scen-t">{titulo}</span>{flip}
                          <div class="scen-n mono">
                            <span style="font-size:13px;color:#8a8d94;text-decoration:line-through">{pct(p)}</span>
                            <span style="font-size:22px;font-weight:500">{pct(p2)}</span>
                            <span style="font-size:13px;color:{cor}">{pp(p2 - p)}</span>
                          </div>
                        </div>""")
                        st.button("Aplicar ao perfil", key=f"apl_{titulo}", on_click=aplicar, args=(mud,))
            else:
                html('<div class="sub" style="padding-top:10px">Rotativo e comprometimento da renda já estão nos níveis de referência.</div>')

    # ── Variáveis técnicas ─────────────────────────────────────
    with st.expander("Variáveis enviadas ao modelo"):
        cols = st.columns(3, gap="large")
        for i, (k, v) in enumerate(features.items()):
            cols[i % 3].markdown(
                f'<div class="frow" title="{k}"><span style="color:#6b6f78">{LABELS.get(k, k)}</span>'
                f'<span class="mono">{fmt(v)}</span></div>',
                unsafe_allow_html=True,
            )

# ── Rodapé ─────────────────────────────────────────────────
links_html = "".join(
    f'<a href="{url}" target="_blank"><img src="https://unpkg.com/lucide-static@0.400.0/icons/{icone}.svg" alt="">{nome}</a>'
    for nome, url, icone in LINKS
)
html(f'<div class="rodape"><span>Desenvolvido por {AUTOR} · Modelo treinado na base Give Me Some Credit</span><nav>{links_html}</nav></div>')
