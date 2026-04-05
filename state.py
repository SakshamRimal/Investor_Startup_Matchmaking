import pandas as pd
from fastapi import HTTPException

from data.dummy_data import INVESTORS, STARTUPS, MATCH_HIST, SECTORS_LIST, STAGES_LIST
from models.compatibility_model import CompatibilityModel
from models.traction_model import TractionModel
from models.sector_model import SectorCompatibilityModel
from models.history_model import InvestmentHistoryModel
from models.suggestion_engine import SuggestionEngine


compatibility_model = CompatibilityModel()
traction_model = TractionModel()
sector_model = SectorCompatibilityModel()
history_model = InvestmentHistoryModel()
suggestion_engine = SuggestionEngine()

print("[1/5] Training CompatibilityModel ...")
r1 = compatibility_model.train(MATCH_HIST)
print(f"      OK AUC-ROC = {r1['auc_roc']}")

print("[2/5] Training TractionModel ...")
r2 = traction_model.train(STARTUPS)
print(f"      OK RMSE = {r2['rmse']}")

print("[3/5] Training SectorCompatibilityModel ...")
r3 = sector_model.train(INVESTORS)
print(f"      OK Method = {r3['method']}  |  Graph edges = {r3['graph_edges']}")

print("[4/5] Training InvestmentHistoryModel ...")
r4 = history_model.train(INVESTORS)
print(f"      OK Val accuracy = {r4['val_accuracy']}")

print("[5/5] Training SuggestionEngine ...")
r5 = suggestion_engine.train(INVESTORS, STARTUPS)
print(f"      OK Clusters = {r5['n_clusters']}  |  Latent dim = {r5['latent_dim']}")

print("\n  All models ready.\n")


def get_investor(investor_id: str) -> pd.Series:
    row = INVESTORS[INVESTORS["investor_id"] == investor_id]
    if row.empty:
        raise HTTPException(404, f"Investor '{investor_id}' not found")
    return row.iloc[0]


def get_startup(startup_id: str) -> pd.Series:
    row = STARTUPS[STARTUPS["startup_id"] == startup_id]
    if row.empty:
        raise HTTPException(404, f"Startup '{startup_id}' not found")
    return row.iloc[0]


def build_match_row(inv: pd.Series, stp: pd.Series) -> dict:
    return {
        "sector_match": int(stp["sector"] in inv["preferred_sectors"]),
        "stage_match": int(stp["stage"] in inv["preferred_stages"]),
        "check_fit": int(inv["min_check"] <= stp["total_raised"] * 0.3 <= inv["max_check"]),
        "growth_good": int(stp["growth_rate"] > 0.10),
        "runway_ok": int(stp["runway_months"] > 6),
        "risk_appetite": inv["risk_appetite"],
        "portfolio_size": inv["portfolio_size"],
        "follow_on_rate": inv["follow_on_rate"],
        "mrr": stp["mrr"],
        "growth_rate": stp["growth_rate"],
        "burn_rate": stp["burn_rate"],
        "runway_months": stp["runway_months"],
        "nps_score": stp["nps_score"],
        "churn_rate": stp["churn_rate"],
        "team_size": stp["team_size"],
        "prior_exits": stp["prior_exits"],
    }
