# app_pronostico.py — v2.3
import streamlit as st
import pandas as pd
import numpy as np
import motor_pronostico as motor

st.set_page_config(page_title="Fútbol Grupo — Pronóstico Profesional", layout="wide")
st.title("⚽ Fútbol Grupo – Pronóstico Profesional")
st.caption(f"HL=300d · Elo K=20 · GSR 0.5 · Ensemble 50/50 (xG reales) · Motor {motor.__version__}")

# =====================
# CARGA DE DATASET
# =====================
@st.cache_data(show_spinner=False)
def load_df_cached(file_name: str, file_bytes: bytes):
    import io
    bio = io.BytesIO(file_bytes)
    bio.name = file_name
    return motor.read_dataset(bio)

st.sidebar.header("Conjunto de datos")
data_file = st.sidebar.file_uploader("Sube tu Excel/CSV actualizado (recomendado: Data History GPT5.xlsx)", type=["xlsx","csv"])
df = None
if data_file is not None:
    df = load_df_cached(data_file.name, data_file.getvalue())
    st.sidebar.success(f"Cargado: {data_file.name} · {len(df):,} filas")

# =====================
# SELECTORES
# =====================
st.sidebar.header("Partido")
if df is not None:
    ligas = sorted(df["league"].dropna().unique().tolist())
    liga = st.sidebar.selectbox("Liga", ligas, index=0 if ligas else None)

    df_liga = df[df["league"] == liga] if liga else pd.DataFrame()
    teams_home = sorted(df_liga["home"].dropna().unique().tolist())
    teams_away = sorted(df_liga["away"].dropna().unique().tolist())

    col1, col2 = st.sidebar.columns(2)
    home = col1.selectbox("Local", teams_home, index=0 if teams_home else None)
    away = col2.selectbox("Visita", teams_away, index=0 if teams_away else None)

    st.sidebar.header("Cuotas (opcionales)")
    l = st.sidebar.number_input("Local", value=2.10, step=0.01, format="%.2f")
    d = st.sidebar.number_input("Empate", value=3.20, step=0.01, format="%.2f")
    v = st.sidebar.number_input("Visita", value=3.80, step=0.01, format="%.2f")

    if st.sidebar.button("🎯 Generar Pronóstico", use_container_width=True):
        res = motor.analyze_match(df, liga, home, away, odds={"home":l,"draw":d,"away":v})

        # Encabezado
        st.subheader(f"{liga} — 🟨 {home} vs 🟩 {away}")

        # 1) 1X2
        p1, px, p2 = res['probs_1x2']['home'], res['probs_1x2']['draw'], res['probs_1x2']['away']
        cj1, cjx, cj2 = res['fair_1x2']['home'], res['fair_1x2']['draw'], res['fair_1x2']['away']
        st.markdown(
            f"**1) 1X2** — 🟨 {p1:.1%} (CJ {cj1}) · ❌ {px:.1%} (CJ {cjx}) · 🟩 {p2:.1%} (CJ {cj2})"
        )

        # 2) Goles esperados
        home_l, away_l = res["lambdas"]["home"], res["lambdas"]["away"]
        st.markdown(f"**2) Goles esperados** — 🟨 {home_l:.2f} · 🟩 {away_l:.2f} · **Total** {home_l+away_l:.2f}")

        # 3) BTTS
        st.markdown(f"**3) BTTS** — Sí {res['btts']['yes']:.1%} · No {res['btts']['no']:.1%}")

        # 4) 1er tiempo (informativo)
        st.markdown("**4) Goles 1er tiempo** — ≥1 gol HT (aprox 65–75% según matchup)")

        # 5) Over/Under Goles (1.5/2.5/3.5)
        st.markdown("**5) Over/Under Goles (1.5, 2.5, 3.5)**")
        ou = res["ou"]
        df_ou = pd.DataFrame(
            [{"Línea": L, "Over": f"{ou[L]['over']*100:.1f}%", "Under": f"{ou[L]['under']*100:.1f}%"} for L in [1.5,2.5,3.5]]
        )
        st.table(df_ou)

        # 6) O/U por EQUIPO — Remates, SOT, Córners y Fouls (70–85%) con líneas .5
        st.markdown("**6) O/U por EQUIPO — Remates, SOT, Córners y Fouls (70–85%)**")
        for metric, title in [("shots","SHOTS"),("sot","SOT"),("corners","CORNERS"),("fouls","FOULS")]:
            lines = res["lines_70_85"].get(metric, [])
            st.markdown(f"**{title}**")
            if lines:
                dfm = pd.DataFrame(lines)
                dfm["Prob."] = (dfm["p"]*100).map(lambda x: f"{x:.0f}%")
                dfm["CJ"] = dfm["cj"].map(lambda x: f"{x:.2f}" if x else None)
                dfm = dfm[["team","line","side","Prob.","CJ"]].rename(columns={"team":"Team","line":"Line","side":"Side"})
                st.table(dfm)
            else:
                st.caption("— Sin líneas en rango 70–85% para esta métrica.")

        # 7) Tarjetas por equipo (tabla)
        st.markdown("**7) Tarjetas por equipo** — modelo base de liga (aprox, informativo)")
        cards = res["cards"]
        df_cards = pd.DataFrame([
            {"Team": home, "FT O1.5": f"{cards['home_ft_o15']*100:.0f}%", "HT O0.5": f"{cards['home_ht_o05']*100:.0f}%"},
            {"Team": away, "FT O1.5": f"{cards['away_ft_o15']*100:.0f}%", "HT O0.5": f"{cards['away_ht_o05']*100:.0f}%"},
        ])
        st.table(df_cards)

        # 8) PI70 / PI80
        st.markdown("**8) PI70 / PI80 (goles totales)**")
        st.write(f"PI70: {res['pi70'][0]}–{res['pi70'][1]}  |  PI80: {res['pi80'][0]}–{res['pi80'][1]}")
        st.caption("Intervalos de probabilidad 70% y 80% (Monte Carlo) para el total de goles.")

        # 9) Bandas ≈50%
        st.markdown("**9) Bandas ≈50%** — líneas cercanas a 50% (goles totales)")
        near50 = pd.DataFrame(res["near50"])
        if not near50.empty:
            near50["Over"] = (near50["over"]*100).map(lambda x: f"{x:.1f}%")
            near50["Under"] = (near50["under"]*100).map(lambda x: f"{x:.1f}%")
            st.table(near50[["line","Over","Under"]].rename(columns={"line":"Línea"}))
        else:
            st.caption("No se identificaron líneas cercanas a 50%.")

        # 10) Mini análisis táctico (descriptivo)
        st.markdown("**10) Mini análisis táctico (descriptivo)**")
        total = res["lambdas"]["home"] + res["lambdas"]["away"]
        bullets = [
            f"**Fuerzas esperadas** — 🟨 {home}: {res['lambdas']['home']:.2f} xG/g; 🟩 {away}: {res['lambdas']['away']:.2f} xG/g; **Total** ~ {total:.2f}.",
            "Tempos esperados: inicio con cautela (0–30’), mayor generación en 31–60’, cierres más verticales 61–90’ (promedio de liga).",
            "Game State Reaction: ajustes moderados (factor 0.5) si hay gol temprano / marca primero.",
            "Localía ya incorporada en las fuerzas por rol (no se aplica ajuste adicional)."
        ]
        st.write("\n".join([f"- {b}" for b in bullets]))

        # 11) Top 5 Picks (70–85% · CJ 1.30–1.50) — mostrando mercado
        st.markdown("**11) 📌 Top 5 Picks (70–85% · CJ 1.30–1.50)**")
        t5 = []
        for x in res["top5"]:
            mercado = x.get("metric","").capitalize()
            equipo = home if x["team"] == home else away
            t5.append({
                "Pick": f"{equipo} {mercado} {x['side']} {x['line']:.1f}",
                "Prob.": f"{x['p']*100:.0f}%",
                "CJ": f"{x['cj']:.2f}" if x["cj"] else None
            })
        st.table(pd.DataFrame(t5))

        # === ANEXO H2H EXTENDIDO ===
        st.markdown("---")
        st.subheader("ANEXO — H2H extendido (no ajusta probabilidades)")
        h2h = res.get("h2h", {})
        if h2h and h2h["n"] > 0:
            sig = "⚠️ Señal Fuerte" if h2h["signal_strong"] else "—"
            wr = f"{h2h['winrate_home_pov']:.0%}" if h2h["winrate_home_pov"] is not None else "–"
            st.write(
                f"n={h2h['n']} (local {h2h['n_home']}, visita {h2h['n_away']}) · "
                f"Winrate {home} (POV actual): {wr} · Rango: {h2h['range']} · {sig}"
            )
            st.caption("Reglas: Half-life H2H=300d; señal fuerte si n≥8, venue≥4/4, |Δ|≥15pp (regla simple).")
        else:
            st.caption("Sin H2H suficiente para este cruce en la liga seleccionada.")
else:
    st.info("Sube primero tu Excel/CSV para habilitar los selectores.")
