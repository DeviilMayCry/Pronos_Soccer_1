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

def read_dataset(path):
    df = pd.read_excel(path, engine="openpyxl")
    df = df.rename(columns={
        "home_team_name": "home",
        "away_team_name": "away",
        "league_name": "league",
        "home_team_goal_count": "home_goals",
        "away_team_goal_count": "away_goals",
        "team_a_xg": "home_xg",
        "team_b_xg": "away_xg",
        "home_team_shots": "home_shots",
        "away_team_shots": "away_shots",
        "home_team_shots_on_target": "home_sot",
        "away_team_shots_on_target": "away_sot",
        "home_team_corner_count": "home_corners",
        "away_team_corner_count": "away_corners",
        "home_team_foul_count": "home_fouls",
        "away_team_foul_count": "away_fouls",
        "home_team_yellow_cards": "home_yellow",
        "away_team_yellow_cards": "away_yellow",
        "home_team_possession": "home_poss",
        "away_team_possession": "away_poss"
    })
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
