from fastapi import APIRouter, HTTPException
import pandas as pd
import state
from schemas import (
    CompatibilityRequest,
    TractionRequest,
    SectorRequest,
    NextSectorRequest,
    SuggestRequest,
)


router = APIRouter()


@router.post("/api/predict_compatibility")
def predict_compatibility(req: CompatibilityRequest):
    inv = state.get_investor(req.investor_id)
    stp = state.get_startup(req.startup_id)
    row = state.build_match_row(inv, stp)

    prob = float(state.compatibility_model.predict(pd.DataFrame([row]))[0])
    label = "Strong Match" if prob >= 0.70 else "Potential Match" if prob >= 0.45 else "Weak Match"

    sector_sim = state.sector_model.investor_sector_fit(
        inv["preferred_sectors"], stp["sector"]
    )

    return {
        "investor_id": req.investor_id,
        "investor_name": inv["name"],
        "startup_id": req.startup_id,
        "startup_name": stp["name"],
        "compatibility_score": round(prob, 4),
        "label": label,
        "sector_similarity": sector_sim,
        "factors": {
            "sector_match": bool(row["sector_match"]),
            "stage_match": bool(row["stage_match"]),
            "check_fit": bool(row["check_fit"]),
            "growth_good": bool(row["growth_good"]),
            "runway_ok": bool(row["runway_ok"]),
        },
    }


@router.post("/api/predict_traction")
def predict_traction(req: TractionRequest):
    stp = state.get_startup(req.startup_id)
    result = state.traction_model.predict_single(stp.to_dict())
    return {
        "startup_id": req.startup_id,
        "startup_name": stp["name"],
        "sector": stp["sector"],
        "stage": stp["stage"],
        **result,
    }


@router.post("/api/sector_similarity")
def sector_similarity(req: SectorRequest):
    if req.sector_a not in state.SECTORS_LIST or req.sector_b not in state.SECTORS_LIST:
        raise HTTPException(400, "Invalid sector name")

    sim = state.sector_model.similarity(req.sector_a, req.sector_b)
    return {
        "sector_a": req.sector_a,
        "sector_b": req.sector_b,
        "similarity_score": sim,
        "top_similar_to_a": state.sector_model.top_similar(req.sector_a, k=3),
        "top_similar_to_b": state.sector_model.top_similar(req.sector_b, k=3),
        "similarity_matrix": state.sector_model.similarity_matrix().to_dict(),
    }


@router.post("/api/predict_next_sector")
def predict_next_sector(req: NextSectorRequest):
    invalid = [s for s in req.history if s not in state.SECTORS_LIST]
    if invalid:
        raise HTTPException(400, f"Unknown sectors: {invalid}")

    predictions = state.history_model.predict_next(req.history, top_k=req.top_k)
    return {
        "history": req.history,
        "predictions": [{"sector": s, "probability": p} for s, p in predictions],
    }


@router.post("/api/suggest")
def suggest_startups(req: SuggestRequest):
    inv = state.get_investor(req.investor_id)
    suggestions = state.suggestion_engine.suggest(
        req.investor_id, top_k=req.top_k, novelty_weight=req.novelty_weight
    )
    return {
        "investor_id": req.investor_id,
        "investor_name": inv["name"],
        "preferred_sectors": inv["preferred_sectors"],
        "novelty_weight": req.novelty_weight,
        "suggestions": suggestions,
    }
