from src.runners.ml_runner import run_ml_evaluation
from src.report import result_to_markdown
from src.db import init_db, save_result


if __name__ == "__main__":

    init_db()

    result = run_ml_evaluation()

    run_id = save_result(result)

    print(result_to_markdown(result))

    print("\nRun ID:", run_id)

    if result.status == "FAIL":
        raise SystemExit(1)