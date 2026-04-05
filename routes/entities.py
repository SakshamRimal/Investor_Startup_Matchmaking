from fastapi import APIRouter
import state


router = APIRouter()


@router.get("/api/investors")
def list_investors():
    inv = state.INVESTORS.copy()
    inv["traction_scores"] = None
    return inv.to_dict(orient="records")


@router.get("/api/startups")
def list_startups():
    scores = state.traction_model.predict(state.STARTUPS)
    df = state.STARTUPS.copy()
    df["traction_score"] = scores.round(2)
    return df.to_dict(orient="records")


@router.get("/api/sectors")
def list_sectors():
    return {"sectors": state.SECTORS_LIST, "stages": state.STAGES_LIST}
