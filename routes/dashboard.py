from fastapi import APIRouter
import state


router = APIRouter()


@router.get("/api/dashboard")
def dashboard():
    scores = state.traction_model.predict(state.STARTUPS)

    top_startups = state.STARTUPS.copy()
    top_startups["traction_score"] = scores
    top5 = (
        top_startups.nlargest(5, "traction_score")
        [["startup_id", "name", "sector", "stage", "traction_score"]]
        .to_dict(orient="records")
    )

    sample_rows = state.MATCH_HIST.sample(min(8, len(state.MATCH_HIST))).copy()
    inv_map = state.INVESTORS.set_index("investor_id")["name"].to_dict()
    stp_map = state.STARTUPS.set_index("startup_id")["name"].to_dict()
    probs = state.compatibility_model.predict(sample_rows)

    recent_matches = []
    for i, (_, row) in enumerate(sample_rows.iterrows()):
        recent_matches.append({
            "investor_id": row["investor_id"],
            "investor_name": inv_map.get(row["investor_id"], row["investor_id"]),
            "startup_id": row["startup_id"],
            "startup_name": stp_map.get(row["startup_id"], row["startup_id"]),
            "score": round(float(probs[i]), 4),
        })
    recent_matches.sort(key=lambda x: -x["score"])

    sector_dist = state.STARTUPS["sector"].value_counts().to_dict()

    return {
        "total_investors": len(state.INVESTORS),
        "total_startups": len(state.STARTUPS),
        "total_matches": len(state.MATCH_HIST),
        "avg_traction": round(float(scores.mean()), 2),
        "top_startups": top5,
        "recent_matches": recent_matches,
        "sector_distribution": sector_dist,
        "model_metrics": {
            "compatibility_auc": state.r1["auc_roc"],
            "traction_rmse": state.r2["rmse"],
            "sector_graph_edges": state.r3["graph_edges"],
            "lstm_val_accuracy": state.r4["val_accuracy"],
        },
    }
