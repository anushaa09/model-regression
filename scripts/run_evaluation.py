from src.runners.ml_runner import run_ml_evaluation
from src.report import result_to_markdown
from src.db import init_db, save_result


if __name__ == "__main__":

    # Initialize database
    init_db()

    # Run ML evaluation
    result = run_ml_evaluation()

    # Save result
    run_id = save_result(result)

    # Print result
    print(result_to_markdown(result))

    print("\nRun ID:", run_id)