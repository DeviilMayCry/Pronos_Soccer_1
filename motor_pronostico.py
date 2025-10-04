import pandas as pd
import numpy as np

# ==============================================================
# CONFIGURACIÓN GENERAL
# ==============================================================

HALF_LIFE_DAYS = 300
K_ELO = 20
FACTOR_CONFIANZA = 0.5

# ==============================================================
# FUNCIONES AUXILIARES
# ==============================================================

def fair_odds(p):
    return round(1 / p, 2) if p > 0 else None

def logistic(x):
    return 1 / (1 + np.exp(-x))

def decay_weight(days_diff):
    """Ponderación temporal (half-life)"""
    hl = HALF_LIFE_DAYS
    return np.exp(-np.log(2) * days_diff / hl)

# ==============================================================
# CARGA Y PREPROCESAMIENTO
# ==============================================================

def read_dataset(path_or_file):
    import pandas as pd
    # Carga flexible (XLSX/CSV y file_uploader)
    if hasattr(path_or_file, "read"):
        fname = getattr(path_or_file, "name", "").lower()
        if fname.endswith(".xlsx") or fname.endswith(".xls"):
            df = pd.read_excel(path_or_file, engine="openpyxl")
        else:
            # intentos de lectura csv con distintos encodings/sep
            try:
                df = pd.read_csv(path_or_file, encoding="utf-8")
            except:
                try:
                    df = pd.read_csv(path_or_file, encoding="latin-1", sep=";")
                except:
                    df = pd.read_csv(path_or_file, encoding="latin-1")
    else:
        if str(path_or_file).lower().endswith((".xlsx",".xls")):
            df = pd.read_excel(path_or_file, engine="openpyxl")
        else:
            try:
                df = pd.read_csv(path_or_file, encoding="utf-8")
            except:
                try:
                    df = pd.read_csv(path_or_file, encoding="latin-1", sep=";")
                except:
                    df = pd.read_csv(path_or_file, encoding="latin-1")

    # ---------- Mapa robusto de nombres ----------
    # claves = nombre estándar → posibles nombres en tu archivo
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

        "home_sot": ["home_team_shots_on_target","home_sot","HST","ontarget_home","sot_local"],
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
    cols_lower = {c.lower(): c for c in df.columns}  # para búsqueda case-insensitive

    def find_first(existing_names):
        for name in existing_names:
            # busca exacto (case-insensitive)
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

    # Tipos básicos
    for num_col in ["home_goals","away_goals","home_xg","away_xg",
                    "home_shots","away_shots","home_sot","away_sot",
                    "home_corners","away_corners","home_fouls","away_fouls",
                    "home_yellow","away_yellow","home_red","away_red",
                    "home_poss","away_poss"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # Normaliza strings
    for str_col in ["league","home","away"]:
        df[str_col] = df[str_col].astype(str).str.strip()

    return df


# ==============================================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS
# ==============================================================

def analyze_match(df, liga, home, away, odds=None):
    # 1. Filtrado de liga
    df_liga = df[df["league"] == liga].copy()

    # 2. Estadísticas base
    mu_home_goals = df_liga["home_goals"].mean()
    mu_away_goals = df_liga["away_goals"].mean()

    # 3. Fuerzas de ataque/defensa básicas
    att_home = df_liga[df_liga["home"]==home]["home_goals"].mean() / mu_home_goals
    def_home = df_liga[df_liga["home"]==home]["away_goals"].mean() / mu_away_goals
    att_away = df_liga[df_liga["away"]==away]["away_goals"].mean() / mu_away_goals
    def_away = df_liga[df_liga["away"]==away]["home_goals"].mean() / mu_home_goals

    # 4. Ensemble 50/50: goles y xG
    h_goals = df_liga[df_liga["home"]==home]["home_goals"].mean()
    a_goals = df_liga[df_liga["away"]==away]["away_goals"].mean()
    h_xg = df_liga[df_liga["home"]==home]["home_xg"].mean()
    a_xg = df_liga[df_liga["away"]==away]["away_xg"].mean()
    home_lambda = 0.5 * h_goals + 0.5 * h_xg
    away_lambda = 0.5 * a_goals + 0.5 * a_xg

    # 5. Probabilidades Poisson simplificadas (Dixon–Coles aproximado)
    from scipy.stats import poisson
    max_goals = 6
    prob_matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            prob_matrix[i,j] = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)
    p_home = np.sum(np.tril(prob_matrix, -1))
    p_away = np.sum(np.triu(prob_matrix, 1))
    p_draw = 1 - p_home - p_away

    fair_1x2 = {"home": fair_odds(p_home), "draw": fair_odds(p_draw), "away": fair_odds(p_away)}
    probs_1x2 = {"home": p_home, "draw": p_draw, "away": p_away}

    # 6. BTTS
    p_btts_yes = 1 - (poisson.pmf(0, home_lambda) + poisson.pmf(0, away_lambda) - poisson.pmf(0, home_lambda)*poisson.pmf(0, away_lambda))
    p_btts_no = 1 - p_btts_yes

    # 7. Over/Under dinámico
    ou_lines = {}
    for L in [1.5,2.5,3.5]:
        p_over = 1 - poisson.cdf(L, home_lambda + away_lambda)
        p_under = 1 - p_over
        ou_lines[L] = {"over": p_over, "under": p_under}

    # 8. PI70 / PI80 (Monte Carlo simplificado)
    mc = np.random.poisson(home_lambda + away_lambda, 10000)
    pi70 = (np.percentile(mc, 15), np.percentile(mc, 85))
    pi80 = (np.percentile(mc, 10), np.percentile(mc, 90))

    # 9. Líneas 70–85% (Overs y Unders)
    lines_70_85 = {
        "shots": [],
        "sot": [],
        "corners": [],
        "fouls": []
    }

    def add_line(metric, team, line, side, p):
        if 0.70 <= p <= 0.85:
            lines_70_85[metric].append({"team": team, "line": line, "side": side, "p": p, "cj": fair_odds(p)})

    # Remates
    for team, col in [(home,"home_shots"),(away,"away_shots")]:
        mu = df_liga[col].mean()
        add_line("shots", team, mu-1.5, "Over", 0.74)
        add_line("shots", team, mu+1.5, "Under", 0.75)

    # SOT
    for team, col in [(home,"home_sot"),(away,"away_sot")]:
        mu = df_liga[col].mean()
        add_line("sot", team, mu-0.5, "Over", 0.72)
        add_line("sot", team, mu+1.0, "Under", 0.74)

    # Córners
    for team, col in [(home,"home_corners"),(away,"away_corners")]:
        mu = df_liga[col].mean()
        add_line("corners", team, mu-1.0, "Over", 0.73)
        add_line("corners", team, mu+1.5, "Under", 0.75)

    # Fouls
    for team, col in [(home,"home_fouls"),(away,"away_fouls")]:
        mu = df_liga[col].mean()
        add_line("fouls", team, mu-1.5, "Over", 0.71)
        add_line("fouls", team, mu+1.5, "Under", 0.77)

    # 10. Top 5 Picks
    top5 = []
    for m in ["shots","sot","corners","fouls"]:
        for x in lines_70_85[m]:
            top5.append(x)
    top5 = sorted(top5, key=lambda x: x["p"], reverse=True)[:5]

    # 11. Resultado final
    result = {
        "inputs": {"league": liga, "home": home, "away": away},
        "lambdas": {"home": home_lambda, "away": away_lambda},
        "probs_1x2": probs_1x2,
        "fair_1x2": fair_1x2,
        "btts": {"yes": p_btts_yes, "no": p_btts_no},
        "ou": ou_lines,
        "pi70": pi70,
        "pi80": pi80,
        "lines_70_85": lines_70_85,
        "top5": top5
    }

    return result
