from datetime import datetime

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime
)

from sqlalchemy.orm import (
    declarative_base,
    sessionmaker
)

from src.config import DB_PATH


# -----------------------------------
# Database connection
# -----------------------------------

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False
    }
)

SessionLocal = sessionmaker(
    bind=engine
)

Base = declarative_base()


# -----------------------------------
# Evaluation Run table
# -----------------------------------

class EvaluationRun(Base):

    __tablename__ = "evaluation_runs"

    id = Column(
        Integer,
        primary_key=True
    )

    module = Column(
        String
    )

    baseline_metric = Column(
        Float
    )

    candidate_metric = Column(
        Float
    )

    p_value = Column(
        Float,
        nullable=True
    )

    status = Column(
        String
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


# -----------------------------------
# Initialize database
# -----------------------------------

def init_db():

    Base.metadata.create_all(
        engine
    )


# -----------------------------------
# Save evaluation result
# -----------------------------------

def save_result(result):

    session = SessionLocal()

    run = EvaluationRun(
        module=result.module,

        baseline_metric=
            result.baseline_metric,

        candidate_metric=
            result.candidate_metric,

        p_value=
            result.p_value,

        status=
            result.status
    )

    session.add(run)

    session.commit()

    session.refresh(run)

    session.close()

    return run.id


# -----------------------------------
# Get evaluation history
# -----------------------------------

def get_history():

    session = SessionLocal()

    runs = (
        session.query(
            EvaluationRun
        )
        .order_by(
            EvaluationRun.created_at.desc()
        )
        .all()
    )

    session.close()

    return runs