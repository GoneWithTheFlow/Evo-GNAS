from sqlalchemy import Column
from sqlalchemy import String, Float
from sqlalchemy.orm import registry, Session
from sqlalchemy import create_engine
from hashlib import sha1
from init_ea import create_db_engine
from train_gnn import Config


mapper_registry = registry()
Base = mapper_registry.generate_base()


# Define the JobSave model for the job_save table.
class JobSave(Base):
    __tablename__ = "job_save"
    pkey = Column(String(41), primary_key=True)
    chip = Column(String, nullable=False)
    job = Column(String, nullable=False)

    def __repr__(self):
        return f"JobSave(pkey={self.pkey!r}, chip={self.chip!r}, job={self.job!r})"


# Create a PostgreSQL engine and initialize database tables.
def create_db_engine():
    engine = create_engine(Config.DB_CONN_STR)
    mapper_registry.metadata.create_all(engine)
    return engine


# Save basic job information to job_save.
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


# Look up the stored graph hash for an architecture id.
def get_hash_by_id(id):
    engine = create_db_engine()
    with Session(engine) as session:
        row = session.execute("select hash from hash where id={}".format(id)).first()
        if row is not None:
            return row[0]
        else:
            return None


# Submit a generated architecture for evaluation by the job recorder.
def request_evaluation(id_list, chip_list=None):
    if chip_list is None:
        chip_list = ["PC_GNN_Evaluation"]
    for id in id_list:
        hash = get_hash_by_id(id)

        for chip in chip_list:
            save_a_job(create_db_engine(), chip, hash)
