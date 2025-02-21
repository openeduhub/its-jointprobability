
import logging
from fastapi import FastAPI
from .logs import setup_logging
from .routers import bayesian_predictions

log = logging.getLogger(__name__)

app = FastAPI(openapi_url="/v3/api-docs")
app.include_router(bayesian_predictions.router)

setup_logging()


@app.get("/_ping", name="Ping")
async def root():
    pass
