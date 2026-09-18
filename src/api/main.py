from fastapi import FastAPI

from src.runners.ml_runner import (
    run_ml_evaluation
)

from src.db import (
    init_db,
    save_result,
    get_history
)


app = FastAPI(
    title="Model Regression Detector"
)


@app.on_event("startup")
def startup():

    init_db()


@app.get("/")
def home():

    return {
        "message":
            "Model Regression Detector API"
    }


@app.post("/run-eval")
def run_evaluation():

    result = run_ml_evaluation()

    run_id = save_result(
        result
    )

    return {
        "run_id": run_id,
        "result": result.model_dump()
    }


@app.get("/history")
def history():

    runs = get_history()

    return [
        {
            "id": run.id,
            "module": run.module,
            "baseline_metric":
                run.baseline_metric,
            "candidate_metric":
                run.candidate_metric,
            "p_value":
                run.p_value,
            "status":
                run.status,
            "created_at":
                run.created_at
        }
        for run in runs
    ]