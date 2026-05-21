from sqlalchemy import Column
from sqlalchemy import String, Float
from sqlalchemy.orm import registry, Session
from sqlalchemy import create_engine
from hashlib import sha1
from database import create_db_engine
from gnn_trainer import Config


mapper_registry = registry()
Base = mapper_registry.generate_base()


# Database model for saved evaluation jobs.
class JobSave(Base):
    __tablename__ = "job_save"
    pkey = Column(String(41), primary_key=True)
    chip = Column(String, nullable=False)
    job = Column(String, nullable=False)

    def __repr__(self):
        return f"JobSave(pkey={self.pkey!r}, chip={self.chip!r}, job={self.job!r})"


# Create the database engine and initialize tables.
def create_db_engine():
    engine = create_engine(Config.DB_CONN_STR)
    mapper_registry.metadata.create_all(engine)
    return engine


# Save a chip evaluation job record.
def save_a_job(engine, chip: str, job: str):
    with Session(engine) as session:
        pkey = sha1((chip + job).encode(encoding="utf-8")).hexdigest()

        if session.query(JobSave).where(JobSave.pkey == pkey).first() is None:
            job_row = JobSave(pkey=pkey, chip=chip, job=job)
            session.add(job_row)
            session.commit()
            return True
        else:
            return False


# Fetch the stored hash for a generated model ID.
def get_hash_by_id(id):
    engine = create_db_engine()
    with Session(engine) as session:
        row = session.execute("select hash from hash where id={}".format(id)).first()
        if row is not None:
            return row[0]
        else:
            return None


# Submit generated model IDs for external evaluation.
def request_evaluation(id_list, chip_list=None):
    if chip_list is None:
        chip_list = ["PC_GNN_Evaluation"]
    for id in id_list:
        hash = get_hash_by_id(id)

        for chip in chip_list:
            save_a_job(create_db_engine(), chip, hash)
