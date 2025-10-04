# app_pronostico.py  (versión con desplegables + nombres “bonitos”)
import streamlit as st
import pandas as pd
import unicodedata
from pathlib import Path
import motor_pronostico as motor

st.set_page_config(page_title="Fútbol Grupo – Pronósticos", layout="wide")

st.title("⚽ Fútbol Grupo – Pronóstico Profesional")
st.caption("HL=300d · Elo K=20 · GSR 0.5 · Ensemble 50/50 (xG reales)")

# ====================== Utilidades de nombres ======================
# Corrige mojibake típico (UTF-8 leído como Latin-1) y normaliza
def fix_mojibake(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip()
    # intenta decodificar mojibake común: "AmÃ©rica" -> "América"
    try:
        s_try = s.encode("latin1").decode("utf-8")
        # si al re-decodificar mejora (tiene acentos válidos), úsalo
        if _looks_better(s, s_try):
            s = s_try
    except Exception:
        pass
    # normaliza forma Unicode
    s = unicodedata.normalize("NFC", s)
    # colapsa espacios
    s = " ".join(s.split())
    return s

def _looks_better(orig: str, candidate: str) -> bool:
    # heurística sencilla: si el candidato contiene letras acentuadas o ñ
    # y el original contiene secuencias típicas de mojibake, lo consideramos mejor
    bad_patterns = ["Ã", "Â", "â", "€", "™"]
    has_bad = any(p in orig for p in bad_patterns)
    has_accents = any(ch in candidate for ch in "áéíóúÁÉÍÓÚñÑçÇäëïöüÄËÏÖÜ")
    return has_bad and has_accents

# Overrides manuales opcionales (por si quieres forzar algún nombre)
# Formato: {"nombre_en_csv_exactamente": "Nombre Bonito"}
MANUAL_OVERRIDES = {
    # Ejemplos:
    # "AmÃ©rica": "América",
    # "Borussia Dortmund ": "Borussia Dortmund",
}

def pretty_name(raw: str) -> str:
    raw = raw if isinstance(raw, str) else str(raw)
    # si definiste override manual, se respeta
    if raw in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[raw]
    # si no, aplicamos reparación automática
    return fix_mojibake(raw)

# Construye mapas de bonito <-> original
def make_pretty_maps(series):
    originals = sorted(set(map(str, series)))
    pretty_map = {orig: pretty_name(orig) for orig in originals}
    # si hubiera colisiones (dos originales que quedan con el mismo "bonito"),
    # resolvemos agregando un sufijo corto para mantener la unicidad visual
    seen = {}
    for k, v in list(pretty_map.items()):
        if v not in seen:
            seen[v] = k
        else:
            # colisión: agrega sufijo corto
            count = 2
            new_v = f"{v} ·{count}"
            while new_v in seen:
                count += 1
                new_v = f"{v} ·{count}"
            pretty_map[k] = new_v
            seen[new_v] = k
    pretty_to_canon = {v: k for k, v in pretty_map.items()}
    return pretty_map, pretty_to_canon

# ====================== Carga de datos ======================
st.sidebar.header("📁 Dataset")
archivo = st.sidebar.file_uploader(
    "Sube tu Excel/CSV actualizado (recomendado: Data History GPT5.xlsx)",
    type=["xlsx", "csv"]
)

if archivo is None:
    st.info("⬆️ Sube tu archivo para comenzar.")
    st.stop()

# Usa la función del motor para renombrar columnas a las esperadas
df = motor.read_dataset(archivo)

# ====================== Desplegables con nombres bonitos ======================
st.sidebar.header("⚽ Partido")

# Ligas (bonito)
ligas_series = df["league"].astype(str)
ligas_pretty_map, ligas_pretty_to_canon = make_pretty_maps(ligas_series)
ligas_pretty = sorted({ligas_pretty_map[x] for x in ligas_series.unique()})

liga_pretty = st.sidebar.selectbox("Liga", ligas_pretty, index=0)
liga = ligas_pretty_to_canon[liga_pretty]

# Filtra por liga elegida
df_l = df[df["league"] == liga].copy()

# Equipos home y away (bonito)
homes_pretty_map, homes_pretty_to_canon = make_pretty_maps(df_l["home"].astype(str))
aways_pretty_map, aways_pretty_to_canon = make_pretty_maps(df_l["away"].astype(str))

homes_pretty = sorted({homes_pretty_map[x] for x in df_l["home"].unique()})
aways_pretty = sorted({aways_pretty_map[x] for x in df_l["away"].unique()})

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

# ====================== Ejecutar análisis ======================
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

    # 5) O/U Goles
    st.markdown("**5) Over/Under Goles (1.5, 2.5, 3.5)**")
    df_ou = pd.DataFrame([
        [L, f"{res['ou'][L]['over']:.1%}", f"{res['ou'][L]['under']:.1%}"]
        for L in [1.5, 2.5, 3.5]
    ], columns=["Línea","Over","Under"])
    st.table(df_ou)

    # 6) O/U por EQUIPO — todas las líneas 70–85%
    st.markdown("**6) O/U por EQUIPO — Remates, SOT, Córners y Fouls (70–85%)**")
    for metric, items in res["lines_70_85"].items():
        if items:
            dfm = pd.DataFrame(items)
            # reemplaza el nombre “crudo” por el bonito para visual
            dfm["team"] = dfm["team"].map(lambda x: home_pretty if x == home else (away_pretty if x == away else x))
            dfm["Prob."] = dfm["p"].map(lambda x: f"{x:.0%}")
            dfm["CJ"] = dfm["cj"].map(lambda x: f"{x:.2f}")
            st.markdown(f"**{metric.upper()}**")
            st.dataframe(dfm[["team","line","side","Prob.","CJ"]], use_container_width=True)

    # 7) Tarjetas
    st.markdown("**7) Tarjetas** — FT O1.5 / HT O0.5 (promedio de liga)")

    # 8) PI70/80
    st.markdown(f"**8) PI70 / PI80** — {res['pi70']} / {res['pi80']} (goles totales)")

    # 9) Bandas ≈50%
    st.markdown("**9) Bandas ≈50%** — líneas con probabilidad cercana a 50%")

    # 10) Mini análisis
    st.markdown("**10) Mini análisis táctico** — timings, GSR y fuerza relativa (descriptivo)")

    # 11) Top 5 Picks
    st.markdown("**11) 📌 Top 5 Picks (70–85% · CJ 1.30–1.50)**")
    t5 = pd.DataFrame([
        {
            "Pick": f"{(home_pretty if x['team']==home else away_pretty)} {x['side']} {x['line']}",
            "Prob.": f"{x['p']:.0%}",
            "CJ": f"{x['cj']:.2f}"
        } for x in res["top5"]
    ])
    st.dataframe(t5, use_container_width=True)

    # Derby / H2H (si lo devuelves en motor)
    if res.get("derby"):
        st.warning(f"⚔️ Derby Detectado: {res['derby']}")

