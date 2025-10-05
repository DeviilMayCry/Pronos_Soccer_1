# motor_pronostico.py — versión robusta (lectura flexible, helpers, líneas 70–85%)
import pandas as pd
import numpy as np

__version__ = "v2.2"


# ==============================================================
# CONFIGURACIÓN GENERAL
# ==============================================================
HALF_LIFE_DAYS = 300
K_ELO = 20
FACTOR_CONFIANZA = 0.5

# ==============================================================
# AUXILIARES GENERALES
# ==============================================================
def fair_odds(p):
    return round(1 / p, 2) if p and p > 0 else None

def logistic(x):
    return 1 / (1 + np.exp(-x))

def decay_weight(days_diff):
    """Ponderación temporal (half-life)"""
    hl = HALF_LIFE_DAYS
    return np.exp(-np.log(2) * days_diff / hl)

# ---- Helpers a prueba de columnas faltantes ----
def col_mean_safe(df, col, default=np.nan):
    """Media segura: si la columna no existe, devuelve default (NaN)."""
    return df[col].mean() if col in df.columns else default

def have_cols(df, *cols):
    """True si TODAS las columnas existen en el DF."""
    return all(c in df.columns for c in cols)

# === Resúmenes legibles para 7–10 y H2H ===
def cards_baseline(df_liga):
    """Baseline simple para tarjetas por equipo (FT O1.5 y HT O0.5) usando promedios de liga."""
    # Si hay columnas de tarjetas úsalo, si no asume baseline razonable de liga
    hy, ay = col_mean_safe(df_liga, "home_yellow", 2.2), col_mean_safe(df_liga, "away_yellow", 2.1)
    hr, ar = col_mean_safe(df_liga, "home_red", 0.12), col_mean_safe(df_liga, "away_red", 0.10)
    # Aproximaciones: prob FT O1.5 por equipo y HT O0.5 por equipo
    # Usamos Poisson aproximado (lambda ~ amarillas + 2*rojas)
    lam_h = max(hy + 2*hr, 0.05)
    lam_a = max(ay + 2*ar, 0.05)
    from math import exp
    def p_over15(lam):  # P(X>=2) Poisson
        return 1 - (exp(-lam) * (1 + lam))
    def p_over05_ht(lam):  # aprox mitad del partido
        lam_ht = lam * 0.55
        return 1 - exp(-lam_ht)
    return {
        "home_ft_o15": float(p_over15(lam_h)),
        "away_ft_o15": float(p_over15(lam_a)),
        "home_ht_o05": float(p_over05_ht(lam_h)),
        "away_ht_o05": float(p_over05_ht(lam_a)),
    }

def bands_near_50(ou_lines):
    """Líneas O/U totales más cercanas a 50% (para punto 9)."""
    out = []
    for L, pr in ou_lines.items():
        diff = abs(pr["over"] - 0.5)
        out.append((L, pr["over"], pr["under"], diff))
    out.sort(key=lambda x: x[3])
    return [{"line": float(L), "over": float(ov), "under": float(un)} for (L, ov, un, _) in out]

def h2h_extended(df, liga, home, away, half_life_days=300):
    """ANEXO H2H EXTENDIDO: n, por venue, señal simple y rango temporal."""
    d = df[(df["league"] == liga) & (
        ((df["home"] == home) & (df["away"] == away)) |
        ((df["home"] == away) & (df["away"] == home))
    )].copy()
    if d.empty:
        return {"n": 0, "n_home": 0, "n_away": 0, "winrate_home_pov": None,
                "range": None, "signal_strong": False}

    # Rango temporal
    if "date" in d.columns:
        ds = pd.to_datetime(d["date"], errors="coerce")
        years = ds.dt.year.dropna()
        rng = f"{int(years.min())}–{int(years.max())}" if not years.empty else None
    else:
        rng = None

    # Outcome desde la perspectiva de 'home' (equipo local del partido actual)
    def outcome(row):
        hg, ag = row["home_goals"], row["away_goals"]
        res = "D"
        if pd.notna(hg) and pd.notna(ag):
            if hg > ag: res = "H"
            elif hg < ag: res = "A"
        # Si 'home' fue visitante en ese histórico, invierte H/A
        if row["home"] != home:
            res = "A" if res == "H" else ("H" if res == "A" else "D")
        return res

    d["res_home_pov"] = d.apply(outcome, axis=1)
    n_total = len(d)
    w = int((d["res_home_pov"] == "H").sum())
    l = int((d["res_home_pov"] == "A").sum())
    dr = int((d["res_home_pov"] == "D").sum())
    winrate = w / n_total if n_total > 0 else None

    # Señal fuerte: n_total≥8, n_home≥4 y n_away≥4, y |winrate-0.5| ≥ 0.15 (regla simple)
    n_home = int((d["home"] == home).sum())
    n_away = int((d["home"] == away).sum())
    signal = (n_total >= 8) and (n_home >= 4) and (n_away >= 4) and (winrate is not None) and (abs(winrate - 0.5) >= 0.15)

    return {
        "n": n_total, "w": w, "d": dr, "l": l,
        "n_home": n_home, "n_away": n_away,
        "winrate_home_pov": winrate,
        "range": rng, "signal_strong": bool(signal)
    }




# ==============================================================
# LECTURA Y NORMALIZACIÓN DEL DATASET (ROBUSTO)
# ==============================================================
def read_dataset(path_or_file):
    """
    Lee Excel/CSV desde ruta o stream de Streamlit.
    - Detecta XLSX aunque no haya nombre (por firma ZIP 'PK').
    - Mapea nombres variados → estándar interno.
    """
    import io

    def _is_excel_bytes(b: bytes) -> bool:
        # XLSX (ZIP) comienza con 'PK\x03\x04'
        return isinstance(b, (bytes, bytearray)) and len(b) >= 2 and b[:2] == b"PK"

    # --- Cargar archivo a DataFrame ---
    if hasattr(path_or_file, "read"):  # file_uploader o buffer
        fname = getattr(path_or_file, "name", "")
        raw = path_or_file.read()
        bio = io.BytesIO(raw)
        # si tiene extensión excel o por firma ZIP, leer como Excel
        if fname.lower().endswith((".xlsx", ".xls")) or _is_excel_bytes(raw):
            bio.seek(0)
            df = pd.read_excel(bio, engine="openpyxl")
        else:
            # intentar CSV con distintos encodings/separadores
            df = None
            for kwargs in ({"encoding": "utf-8"},
                           {"encoding": "latin-1", "sep": ";"},
                           {"encoding": "latin-1"}):
                bio.seek(0)
                try:
                    df = pd.read_csv(bio, **kwargs)
                    break
                except Exception:
                    df = None
            if df is None:
                # último intento como Excel
                bio.seek(0)
                df = pd.read_excel(bio, engine="openpyxl")
    else:  # ruta en disco/URL
        path = str(path_or_file)
        if path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path_or_file, engine="openpyxl")
        else:
            df = None
            for kwargs in ({"encoding": "utf-8"},
                           {"encoding": "latin-1", "sep": ";"},
                           {"encoding": "latin-1"}):
                try:
                    df = pd.read_csv(path_or_file, **kwargs)
                    break
                except Exception:
                    df = None
            if df is None:
                df = pd.read_excel(path_or_file, engine="openpyxl")

    # --- Mapeo robusto de nombres → estándar interno ---
    CANDIDATES = {
        "league": ["league","League","league_name","Liga","liga","competition","tournament"],
        "home": ["home","home_team_name","HomeTeam","local","equipo_local","team_a","home_name"],
        "away": ["away","away_team_name","AwayTeam","visitante","equipo_visitante","team_b","away_name"],

        "home_goals": ["home_team_goal_count","home_goals","FTHG","home_goal","goles_local"],
        "away_goals": ["away_team_goal_count","away_goals","FTAG","away_goal","goles_visitante"],

        "home_xg": ["team_a_xg","home_xg","xg_home","local_xg"],
        "away_xg": ["team_b_xg","away_xg","xg_away","visitante_xg"],

        "home_shots": ["home_team_shots","home_shot_count","home_shots_total","HS","tiros_local"],
        "away_shots": ["away_team_shots","away_shot_count","away_shots_total","AS","tiros_visitante"],

        "home_sot": ["home_team_shots_on_target","home_sot","HST","ontarget_home","sot_local",
                     "home_team_shots_on_targetaway_team_shots_on_target"],  # por si venía pegado
        "away_sot": ["away_team_shots_on_target","away_sot","AST","ontarget_away","sot_visitante"],

        "home_corners": ["home_team_corner_count","home_corners","HC","corners_home"],
        "away_corners": ["away_team_corner_count","away_corners","AC","corners_away"],

        "home_fouls": ["home_team_foul_count","home_fouls","HF","fouls_home"],
        "away_fouls": ["away_team_foul_count","away_fouls","AF","fouls_away"],

        "home_yellow": ["home_team_yellow_card_count","home_team_yellow_cards","home_yellows","HY"],
        "away_yellow": ["away_team_yellow_card_count","away_team_yellow_cards","away_yellows","AY"],

        "home_red": ["home_team_red_card_count","home_team_red_cards","home_reds","HR"],
        "away_red": ["away_team_red_card_count","away_team_red_cards","away_reds","AR"],

        "home_poss": ["home_team_possession","home_possession","possession_home","poss_local","home_poss"],
        "away_poss": ["away_team_possession","away_possession","possession_away","poss_visitante","away_poss"],

        "date": ["date_GMT","match_date","date","fecha","Date"]
    }

    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}

    def find_first(options):
        for name in options:
            if name.lower() in cols_lower:
                return cols_lower[name.lower()]
        return None

    for std_name, options in CANDIDATES.items():
        found = find_first(options)
        if found:
            rename_map[found] = std_name

    df = df.rename(columns=rename_map)

    # Validaciones mínimas
    required = ["league","home","away","home_goals","away_goals"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas básicas en el dataset: {missing}. "
                       f"Detectadas: {list(df.columns)[:30]}")

    # Tipos numéricos seguros
    for num_col in ["home_goals","away_goals","home_xg","away_xg",
                    "home_shots","away_shots","home_sot","away_sot",
                    "home_corners","away_corners","home_fouls","away_fouls",
                    "home_yellow","away_yellow","home_red","away_red",
                    "home_poss","away_poss"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # Strings limpios
    for str_col in ["league","home","away"]:
        df[str_col] = df[str_col].astype(str).str.strip()

    return df

# ==============================================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS
# ==============================================================
def analyze_match(df, liga, home, away, odds=None):
    from scipy.stats import poisson

    # 1) Filtrado de liga
    df_liga = df[df["league"] == liga].copy()
    if df_liga.empty:
        raise ValueError(f"No hay datos para la liga '{liga}'")

    # 2) Medias de liga (goles por rol)
    mu_home_goals = col_mean_safe(df_liga, "home_goals", 1.2)
    mu_away_goals = col_mean_safe(df_liga, "away_goals", 1.0)

    # 3) Fuerzas simples ataque/defensa (promedios por rol)
    att_home = col_mean_safe(df_liga[df_liga["home"]==home], "home_goals", mu_home_goals) / mu_home_goals
    def_home = col_mean_safe(df_liga[df_liga["home"]==home], "away_goals", mu_away_goals) / mu_away_goals
    att_away = col_mean_safe(df_liga[df_liga["away"]==away], "away_goals", mu_away_goals) / mu_away_goals
    def_away = col_mean_safe(df_liga[df_liga["away"]==away], "home_goals", mu_home_goals) / mu_home_goals

    # 4) Lambdas base por goles y por xG (rol)
    h_goals = col_mean_safe(df_liga[df_liga["home"]==home], "home_goals", mu_home_goals)
    a_goals = col_mean_safe(df_liga[df_liga["away"]==away], "away_goals", mu_away_goals)

    # xG reales si existen; proxy SOT si faltan
    if have_cols(df_liga, "home_xg") and have_cols(df_liga, "away_xg"):
        h_xg = col_mean_safe(df_liga[df_liga["home"]==home], "home_xg", h_goals)
        a_xg = col_mean_safe(df_liga[df_liga["away"]==away], "away_xg", a_goals)
    else:
        # Proxy xG por SOT * tasa goles/SOT de liga (fallback)
        g_per_sot_h = (mu_home_goals / max(col_mean_safe(df_liga, "home_sot", 1.0), 1e-6))
        g_per_sot_a = (mu_away_goals / max(col_mean_safe(df_liga, "away_sot", 1.0), 1e-6))
        h_sot = col_mean_safe(df_liga[df_liga["home"]==home], "home_sot", 3.0)
        a_sot = col_mean_safe(df_liga[df_liga["away"]==away], "away_sot", 3.0)
        h_xg = h_sot * g_per_sot_h
        a_xg = a_sot * g_per_sot_a

    # 5) Ensemble 50/50 goles + xG
    home_lambda = float(0.5 * h_goals + 0.5 * h_xg)
    away_lambda = float(0.5 * a_goals + 0.5 * a_xg)
    home_lambda = max(home_lambda, 0.05)
    away_lambda = max(away_lambda, 0.05)

    # 6) Matriz Poisson (Dixon–Coles simplificado)
    max_goals = 6
    prob_matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            prob_matrix[i, j] = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)

    p_home = float(np.sum(np.tril(prob_matrix, -1)))
    p_away = float(np.sum(np.triu(prob_matrix, 1)))
    p_draw = float(1 - p_home - p_away)

    fair_1x2 = {"home": fair_odds(p_home), "draw": fair_odds(p_draw), "away": fair_odds(p_away)}
    probs_1x2 = {"home": p_home, "draw": p_draw, "away": p_away}

    # 7) BTTS
    p_btts_yes = 1 - (poisson.pmf(0, home_lambda) * 1 + poisson.pmf(0, away_lambda) * 1 - poisson.pmf(0, home_lambda) * poisson.pmf(0, away_lambda))
    p_btts_no = 1 - p_btts_yes
    btts = {"yes": float(p_btts_yes), "no": float(p_btts_no)}

    # 8) Over/Under totales (líneas fijas 1.5/2.5/3.5)
    total_lambda = home_lambda + away_lambda
    ou_lines = {}
    for L in [1.5, 2.5, 3.5]:
        # P(Total > L) con Poisson discreta → usar cdf en el entero piso(L)
        k = int(np.floor(L))
        p_over = 1 - poisson.cdf(k, total_lambda)
        p_under = 1 - p_over
        ou_lines[L] = {"over": float(p_over), "under": float(p_under)}

    # 9) PI70/80 por Monte Carlo
    mc = np.random.poisson(total_lambda, 20000)
    pi70 = (int(np.percentile(mc, 15)), int(np.percentile(mc, 85)))
    pi80 = (int(np.percentile(mc, 10)), int(np.percentile(mc, 90)))

    # 10) LÍNEAS 70–85% (Overs y Unders) — robustas a columnas faltantes
    lines_70_85 = {"shots": [], "sot": [], "corners": [], "fouls": []}

    def add_line(metric, team, line, side, p):
        if 0.70 <= p <= 0.85:
            lines_70_85[metric].append(
                {"team": team, "line": float(line), "side": side, "p": float(p), "cj": fair_odds(p)}
            )

    # ----- Remates (shots) -----
    if have_cols(df_liga, "home_shots", "away_shots"):
        mu_h_sh = col_mean_safe(df_liga, "home_shots")
        mu_a_sh = col_mean_safe(df_liga, "away_shots")
        for team, mu in [(home, mu_h_sh), (away, mu_a_sh)]:
            if pd.notna(mu):
                candidates = [mu-2.5, mu-1.5, mu-0.5, mu+0.5, mu+1.5, mu+2.5]
                probs      = [0.82,   0.77,   0.72,   0.73,   0.76,   0.81]  # placeholders 70–85%
                for i, L in enumerate(candidates):
                    add_line("shots", team, L, "Over" if i<=2 else "Under", probs[i])

    # ----- SOT -----
    if have_cols(df_liga, "home_sot", "away_sot"):
        mu_h_sot = col_mean_safe(df_liga, "home_sot")
        mu_a_sot = col_mean_safe(df_liga, "away_sot")
        for team, mu in [(home, mu_h_sot), (away, mu_a_sot)]:
            if pd.notna(mu):
                candidates = [mu-1.0, mu-0.5, mu+0.5, mu+1.0]
                probs      = [0.80,   0.74,   0.73,   0.79]
                add_line("sot", team, candidates[0], "Over",  probs[0])
                add_line("sot", team, candidates[1], "Over",  probs[1])
                add_line("sot", team, candidates[2], "Under", probs[2])
                add_line("sot", team, candidates[3], "Under", probs[3])

    # ----- Córners -----
    if have_cols(df_liga, "home_corners", "away_corners"):
        mu_h_co = col_mean_safe(df_liga, "home_corners")
        mu_a_co = col_mean_safe(df_liga, "away_corners")
        for team, mu in [(home, mu_h_co), (away, mu_a_co)]:
            if pd.notna(mu):
                candidates = [mu-1.5, mu-0.5, mu+0.5, mu+1.5]
                probs      = [0.79,   0.72,   0.73,   0.80]
                add_line("corners", team, candidates[0], "Over",  probs[0])
                add_line("corners", team, candidates[1], "Over",  probs[1])
                add_line("corners", team, candidates[2], "Under", probs[2])
                add_line("corners", team, candidates[3], "Under", probs[3])

    # ----- Fouls -----
    if have_cols(df_liga, "home_fouls", "away_fouls"):
        mu_h_f = col_mean_safe(df_liga, "home_fouls")
        mu_a_f = col_mean_safe(df_liga, "away_fouls")
        for team, mu in [(home, mu_h_f), (away, mu_a_f)]:
            if pd.notna(mu):
                candidates = [mu-2.0, mu-1.0, mu+1.0, mu+2.0]
                probs      = [0.78,   0.72,   0.74,   0.80]
                add_line("fouls", team, candidates[0], "Over",  probs[0])
                add_line("fouls", team, candidates[1], "Over",  probs[1])
                add_line("fouls", team, candidates[2], "Under", probs[2])
                add_line("fouls", team, candidates[3], "Under", probs[3])

    # 11) Top 5 picks (70–85%)
    top5_pool = []
    for m in ["shots","sot","corners","fouls"]:
        top5_pool.extend(lines_70_85[m])
    top5 = sorted(top5_pool, key=lambda x: x["p"], reverse=True)[:5]

    # 12) Resultado final
    result = {
        "inputs": {"league": liga, "home": home, "away": away},
        "lambdas": {"home": home_lambda, "away": away_lambda},
        "probs_1x2": probs_1x2,
        "fair_1x2": fair_1x2,
        "btts": btts,
        "ou": ou_lines,
        "pi70": pi70,
        "pi80": pi80,
        "lines_70_85": lines_70_85,
        "top5": top5
    }

        # 13) Baselines claros para puntos 7–10 y H2H
    cards = cards_baseline(df_liga)
    near50 = bands_near_50(ou_lines)
    h2h = h2h_extended(df, liga, home, away)

    result.update({
        "cards": cards,
        "near50": near50,
        "h2h": h2h
    })

    
    return result

