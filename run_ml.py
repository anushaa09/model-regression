from src.runners.ml_runner import run_ml_evaluation


result = run_ml_evaluation()

print(
    result.model_dump_json(
        indent=2
    )
)