from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import (
    entities_router,
    predictions_router,
    dashboard_router,
    frontend_router,
)


app = FastAPI(
    title="InvestMatch Pro",
    description="AI-Powered Startup-Investor Matchmaking Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(entities_router)
app.include_router(predictions_router)
app.include_router(dashboard_router)
app.include_router(frontend_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
