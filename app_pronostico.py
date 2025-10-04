import streamlit as st
import pandas as pd
from pathlib import Path
import motor_pronostico as motor

st.set_page_config(page_title="Fútbol Grupo – Pronósticos", layout="wide")

st.title("⚽ Fútbol Grupo – Pronóstico Profesional")
st.caption("HL=300d · Elo K=20 · GSR 0.5 · Ensemble 50/50 (xG reales)")

# ======== Dataset ========
st.sidebar.header("📁 Dataset")
archivo = st.sidebar.file_uploader("Sube tu Excel/CSV actualizado",
                                   type=["xlsx","csv"])

if archivo is None:
    st.info("⬆️ Sube tu archivo para comenzar.")
    st.stop()

df = motor.read_dataset(archivo)

# ======== Selección del partido ========
st.sidebar.header("⚽ Partido")
ligas = sorted(df["league"].astype(str).unique())
liga = st.sidebar.selectbox("Liga", ligas)
df_l = df[df["league"] == liga]
home = st.sidebar.selectbox("Equipo Local", sorted(df_l["home"].unique()))
away = st.sidebar.selectbox("Equipo Visitante", sorted(df_l["away"].unique()))
if home == away:
    st.sidebar.error("Elige equipos distintos.")
    st.stop()

st.sidebar.header("💸 Cuotas (opcionales)")
oh = st.sidebar.number_input("Local", value=2.50)
od = st.sidebar.number_input("Empate", value=3.20)
oa = st.sidebar.number_input("Visita", value=2.60)
odds = {"home": oh, "draw": od, "away": oa}

# ======== Botón para análisis ========
if st.sidebar.button("🚀 Generar Pronóstico", use_container_width=True):
    with st.spinner("Calculando..."):
        res = motor.analyze_match(df, liga, home, away, odds)

    st.subheader(f"{liga} — 🟨 {home} vs 🟩 {away}")

    st.markdown(f"**1) 1X2:** {res['probs_1x2']} (CJ={res['fair_1x2']})")
    st.markdown(f"**2) Goles esperados:** 🟨 {res['lambdas']['home']:.2f} · 🟩 {res['lambdas']['away']:.2f}")

    st.markdown(f"**3) BTTS:** Sí {res['btts']['yes']:.2%} / No {res['btts']['no']:.2%}")
    st.markdown("**4) Goles 1er tiempo:** estimación 65–75% de ≥1 gol HT")

    st.markdown("**5) Over/Under Goles (1.5, 2.5, 3.5)**")
    df_ou = pd.DataFrame([
        [L, f"{res['ou'][L]['over']:.2%}", f"{res['ou'][L]['under']:.2%}"]
        for L in [1.5, 2.5, 3.5]
    ], columns=["Línea","Over","Under"])
    st.table(df_ou)

    st.markdown("**6) O/U por EQUIPO — Remates, SOT, Córners, Fouls (70–85%)**")
    for metric, items in res["lines_70_85"].items():
        if items:
            dfm = pd.DataFrame(items)
            dfm["Prob."] = dfm["p"].map(lambda x: f"{x:.2%}")
            dfm["CJ"] = dfm["cj"].map(lambda x: f"{x:.2f}")
            st.dataframe(dfm[["team","line","side","Prob.","CJ"]], use_container_width=True)

    st.markdown(f"**7) Tarjetas:** FT O1.5 / HT O0.5 (promedio de liga)")
    st.markdown(f"**8) PI70 / PI80:** {res['pi70']} / {res['pi80']}")

    st.markdown("**9) Bandas ≈50%:** visibles en métricas sensibles (goles, tiros).")
    st.markdown("**10) Mini análisis táctico:** timings + reacciones (descr.)")

    st.markdown("**11) 📌 Top 5 Picks (70–85%)**")
    t5 = pd.DataFrame([{"Pick": f"{x['team']} {x['side']} {x['line']} ({x['p']:.0%})", "CJ": x["cj"]} for x in res["top5"]])
    st.table(t5)
