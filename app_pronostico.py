# app_pronostico.py  — versión rápida (cache) + nombres bonitos + desplegables
import streamlit as st
import pandas as pd
import unicodedata
import io, hashlib
from pathlib import Path
import motor_pronostico as motor

st.caption(f"HL=300d · Elo K=20 · GSR 0.5 · Ensemble 50/50 (xG reales) · Motor {motor.__version__}")
st.set_page_config(page_title="Fútbol Grupo – Pronósticos", layout="wide")

st.title("⚽ Fútbol Grupo – Pronóstico Profesional")
st.caption("HL=300d · Elo K=20 · GSR 0.5 · Ensemble 50/50 (xG reales)")

# ====================== Utilidades de nombres ======================
def _looks_better(orig: str, candidate: str) -> bool:
    bad_patterns = ["Ã", "Â", "â", "€", "™"]
    has_bad = any(p in orig for p in bad_patterns)
    has_accents = any(ch in candidate for ch in "áéíóúÁÉÍÓÚñÑçÇäëïöüÄËÏÖÜ")
    return has_bad and has_accents

def fix_mojibake(s: str) -> str:
    if not isinstance(s, str): return s
    s = s.strip()
    try:
        s_try = s.encode("latin1").decode("utf-8")
        if _looks_better(s, s_try):
            s = s_try
    except Exception:
        pass
    s = unicodedata.normalize("NFC", s)
    s = " ".join(s.split())
    return s

# Overrides manuales (opcional): {"nombre_exactamente_en_csv": "Nombre Bonito"}
MANUAL_OVERRIDES = {
    # "AmÃ©rica": "América",
}

def pretty_name(raw: str) -> str:
    raw = raw if isinstance(raw, str) else str(raw)
    if raw in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[raw]
    return fix_mojibake(raw)

def make_pretty_maps(series):
    originals = sorted(set(map(str, series)))
    pretty_map = {orig: pretty_name(orig) for orig in originals}
    seen = {}
    for k, v in list(pretty_map.items()):
        if v not in seen:
            seen[v] = k
        else:
            count = 2
            new_v = f"{v} ·{count}"
            while new_v in seen:
                count += 1
                new_v = f"{v} ·{count}"
            pretty_map[k] = new_v
            seen[new_v] = k
    pretty_to_canon = {v: k for k, v in pretty_map.items()}
    return pretty_map, pretty_to_canon

# ====================== Caché: lectura de dataset ======================
def _hash_file(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    return hashlib.md5(data).hexdigest()

@st.cache_data(show_spinner=False)
def load_df_cached(file_name: str, file_bytes: bytes):
    import io
    bio = io.BytesIO(file_bytes)
    bio.name = file_name  # ✅ preserva extensión (.xlsx / .csv)
    return motor.read_dataset(bio)  # el motor ya normaliza columnas


# ====================== Caché: ligas/equipos bonitos ======================
@st.cache_data(show_spinner=False)
def build_league_maps(df_leagues_col):
    ligas_series = df_leagues_col.astype(str)
    ligas_pretty_map, ligas_pretty_to_canon = make_pretty_maps(ligas_series)
    ligas_pretty = sorted({ligas_pretty_map[x] for x in ligas_series.unique()})
    return ligas_pretty, ligas_pretty_map, ligas_pretty_to_canon

@st.cache_data(show_spinner=False)
def build_team_maps(df, liga):
    df_l = df[df["league"] == liga].copy()
    homes_pretty_map, homes_pretty_to_canon = make_pretty_maps(df_l["home"].astype(str))
    aways_pretty_map, aways_pretty_to_canon = make_pretty_maps(df_l["away"].astype(str))
    homes_pretty = sorted({homes_pretty_map[x] for x in df_l["home"].unique()})
    aways_pretty = sorted({aways_pretty_map[x] for x in df_l["away"].unique()})
    return df_l, homes_pretty, aways_pretty, homes_pretty_map, homes_pretty_to_canon, aways_pretty_map, aways_pretty_to_canon

# ====================== Dataset (sidebar) ======================
st.sidebar.header("📁 Conjunto de datos")
archivo = st.sidebar.file_uploader(
    "Sube tu Excel/CSV actualizado (recomendado: Data History GPT5.xlsx)",
    type=["xlsx", "csv"]
)

if archivo is None:
    st.info("⬆️ Sube tu archivo para comenzar.")
    st.stop()

# Solo leemos una vez gracias al caché
file_hash = _hash_file(archivo)
df = load_df_cached(archivo.name, archivo.getvalue())

# ====================== Partido (desplegables) ======================
st.sidebar.header("⚽ Partido")

ligas_pretty, ligas_pretty_map, ligas_pretty_to_canon = build_league_maps(df["league"])
liga_pretty = st.sidebar.selectbox("Liga", ligas_pretty, index=0)
liga = ligas_pretty_to_canon[liga_pretty]

df_l, homes_pretty, aways_pretty, homes_pretty_map, homes_pretty_to_canon, aways_pretty_map, aways_pretty_to_canon = build_team_maps(df, liga)

c1, c2 = st.sidebar.columns(2)
home_pretty = c1.selectbox("Local", homes_pretty, index=0)
away_pretty = c2.selectbox("Visita", aways_pretty, index=0)

home = homes_pretty_to_canon[home_pretty]
away = aways_pretty_to_canon[away_pretty]

if home == away:
    st.sidebar.error("Elige equipos distintos.")
    st.stop()

st.sidebar.header("💸 Cuotas (opcionales)")
oh = st.sidebar.number_input("Local", value=2.50, min_value=1.01, step=0.01)
od = st.sidebar.number_input("Empate", value=3.20, min_value=1.01, step=0.01)
oa = st.sidebar.number_input("Visita", value=2.60, min_value=1.01, step=0.01)
odds = {"home": float(oh), "draw": float(od), "away": float(oa)}

# ====================== Ejecutar análisis solo al pulsar ======================
def pct(x):
    try: return f"{100*float(x):.1f}%"
    except: return "-"

if st.sidebar.button("🚀 Generar Pronóstico", use_container_width=True):
    with st.spinner("Calculando…"):
        res = motor.analyze_match(df, liga, home, away, odds)

    # Encabezado con nombres bonitos
    st.subheader(f"{liga_pretty} — 🟨 {home_pretty} vs 🟩 {away_pretty}")

    # 1) 1X2
    p1 = res["probs_1x2"]; f1 = res["fair_1x2"]
    st.markdown(
        f"**1) 1X2** — "
        f"🟨 {pct(p1['home'])} (CJ {f1['home']}) · "
        f"X {pct(p1['draw'])} (CJ {f1['draw']}) · "
        f"🟩 {pct(p1['away'])} (CJ {f1['away']})"
    )

    # 2) Goles esperados (Ensemble)
    lh, la = res["lambdas"]["home"], res["lambdas"]["away"]
    st.markdown(f"**2) Goles esperados** — 🟨 {lh:.2f} · 🟩 {la:.2f} · **Total {lh+la:.2f}**")

    # 3) BTTS
    btts = res["btts"]
    st.markdown(f"**3) BTTS** — Sí {pct(btts['yes'])} / No {pct(btts['no'])}")

    # 4) HT
    st.markdown("**4) Goles 1er tiempo** — ≥1 gol HT (aprox 65–75% según matchup)")

    # 5) O/U Goles fijo 1.5/2.5/3.5
    st.markdown("**5) Over/Under Goles (1.5, 2.5, 3.5)**")
    df_ou = pd.DataFrame([
        [L, f"{res['ou'][L]['over']:.1%}", f"{res['ou'][L]['under']:.1%}"]
        for L in [1.5, 2.5, 3.5]
    ], columns=["Línea","Over","Under"])
    st.table(df_ou)

    # 6) O/U por EQUIPO — TODAS las líneas 70–85%
    st.markdown("**6) O/U por EQUIPO — Remates, SOT, Córners y Fouls (70–85%)**")
    for metric, items in res.get("lines_70_85", {}).items():
        if not items:
            continue
        dfm = pd.DataFrame(items)
        dfm["team"] = dfm["team"].map(lambda x: home_pretty if x == home else (away_pretty if x == away else x))
        dfm["Prob."] = dfm["p"].map(lambda x: f"{x:.0%}")
        dfm["CJ"] = dfm["cj"].map(lambda x: f"{x:.2f}")
        st.markdown(f"**{metric.upper()}**")
        st.dataframe(dfm[["team","line","side","Prob.","CJ"]], use_container_width=True)

        # 7) Tarjetas (claro)
    st.markdown("**7) Tarjetas por equipo** — modelo base de liga (aprox, informativo)")
    c_home = res["cards"]["home_ft_o15"]; c_away = res["cards"]["away_ft_o15"]
    ch_ht  = res["cards"]["home_ht_o05"]; ca_ht  = res["cards"]["away_ht_o05"]
    st.write(
        f"🟨 {home_pretty} — **FT O1.5**: {c_home:.0%} · **HT O0.5**: {ch_ht:.0%} | "
        f"🟩 {away_pretty} — **FT O1.5**: {c_away:.0%} · **HT O0.5**: {ca_ht:.0%}"
    )

    # 8) PI70 / PI80 explicado
    st.markdown("**8) PI70 / PI80 (goles totales)**")
    st.write(f"PI70: {res['pi70'][0]}–{res['pi70'][1]}  |  PI80: {res['pi80'][0]}–{res['pi80'][1]}")
    st.caption("Intervalos de probabilidad sobre el total de goles: 70% y 80% respectivamente (Monte Carlo).")

    # 9) Bandas ≈50% por métrica (usamos totals como referencia)
    st.markdown("**9) Bandas ≈50%** — líneas cercanas a 50% (goles totales)")
    df50 = pd.DataFrame(res["near50"])
    if not df50.empty:
        df50["Over"] = (df50["over"]*100).map(lambda x: f"{x:.1f}%")
        df50["Under"] = (df50["under"]*100).map(lambda x: f"{x:.1f}%")
        st.table(df50[["line","Over","Under"]].rename(columns={"line":"Línea"}))
    else:
        st.caption("No se identificaron líneas cercanas a 50%.")

    # 10) Mini análisis táctico extendido
    st.markdown("**10) Mini análisis táctico (descriptivo)**")
    total = res["lambdas"]["home"] + res["lambdas"]["away"]
    bullets = [
        f"**Fuerzas esperadas** — 🟨 {home_pretty}: {res['lambdas']['home']:.2f} xG/g; "
        f"🟩 {away_pretty}: {res['lambdas']['away']:.2f} xG/g; **Total** ~ {total:.2f}.",
        "Tempos esperados: inicio con cautela (0–30'), mejora de generación en 31–60', y cierres más verticales 61–90' (promedio de liga).",
        "Game State Reaction: ajustes moderados (factor 0.5) si hay gol temprano / marca primero.",
        "Localía ya incorporada en las fuerzas por rol (no se aplica ajuste adicional)."
    ]
    st.write("\n".join([f"- {b}" for b in bullets]))

    # 11) Top 5 Picks (ya lo tienes arriba, lo dejamos)
    st.markdown("**11) 📌 Top 5 Picks (70–85% · CJ 1.30–1.50)**")
    t5 = pd.DataFrame([
        {
            "Pick": f"{(home_pretty if x['team']==home else away_pretty)} {x['side']} {x['line']:.1f}",
            "Prob.": f"{x['p']:.0%}",
            "CJ": f"{x['cj']:.2f}"
        } for x in res["top5"]
    ])
    st.dataframe(t5, use_container_width=True)

    # === ANEXO H2H EXTENDIDO ===
    st.markdown("---")
    st.markdown("### ANEXO — H2H extendido (no ajusta probabilidades)")
    h2h = res.get("h2h", {})
    if h2h and h2h["n"] > 0:
        sig = "⚠️ Señal Fuerte" if h2h["signal_strong"] else "—"
        st.write(
            f"n={h2h['n']} (local {h2h['n_home']}, visita {h2h['n_away']}) · "
            f"Winrate {home_pretty} (pov actual): {h2h['winrate_home_pov']:.0%} · "
            f"Rango: {h2h['range']} · {sig}"
        )
        st.caption("Reglas: Half-life H2H=300d; señal fuerte si n≥8, venue≥4/4, |Δ|≥15pp (regla simple).")
    else:
        st.caption("Sin H2H suficiente para este cruce en la liga seleccionada.")

