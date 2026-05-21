from sqlalchemy.orm import registry, Session
from sqlalchemy import Column, String, Float, Integer,delete
from sqlalchemy import create_engine
from gnn_trainer import Config


mapper_registry = registry()


Base = mapper_registry.generate_base()


# Database model for base gene records.
class Main(Base):
    __tablename__ = "main"
    id = Column(Integer, primary_key=True)
    gene = Column(String, nullable=False)

    def __repr__(self):
        return f"Main(id={self.id!r}, gene={self.gene!r})"


# Database model for architecture scores.
class Score(Base):
    __tablename__ = "score"
    id = Column(Integer, primary_key=True)
    score = Column(Float, nullable=False)

    def __repr__(self):
        return f"Score(id={self.id!r}, score={self.score!r})"


# Database model for training runtime records.
class Runtime(Base):
    __tablename__ = "runtime"
    id = Column(Integer, primary_key=True)
    runtime = Column(Float, nullable=False)

    def __repr__(self):
        return f"Runtime(id={self.id!r}, runtime={self.runtime!r})"


# Database model for task accuracy metrics.
class Acc(Base):
    __tablename__ = 'acc'
    id = Column(Integer, primary_key=True)
    accuracy = Column(Float, nullable=False)
    mis_size = Column(Float, nullable=False)

    def __repr__(self):
        return f"Acc(id={self.id!r}, accuracy={self.accuracy!r}, mis_size={self.mis_size!r})"


# Database model for generation lineage.
class Generation(Base):
    __tablename__ = "generation"
    id = Column(Integer, primary_key=True)
    father = Column(Integer, nullable=False)
    mother = Column(Integer, nullable=False)
    iteration = Column(Integer, nullable=False)

    def __repr__(self):
        return f"Generation(id={self.id!r}, father={self.father!r}, mother={self.mother!r}, iteration={self.iteration!r})"


# Database model for gene lengths.
class Length(Base):
    __tablename__ = "length"
    id = Column(Integer, primary_key=True)
    length = Column(Integer, nullable=False)
    gene_code_length = Column(Integer, nullable=False)

    def __repr__(self):
        return f"Length(id={self.id!r}, length={self.length!r}, gene_code_length={self.gene_code_length!r})"


# Database model for iteration counters.
class Number(Base):
    __tablename__ = "number"
    id = Column(Integer, primary_key=True)
    number = Column(Integer, nullable=False)
    avg_length = Column(Integer, nullable=False)

    def __repr__(self):
        return f"Number(id={self.id!r}, number={self.number!r}, avg_length={self.avg_length!r})"


# Database model for generated source code.
class Code(Base):
    __tablename__ = "code"
    id = Column(Integer, primary_key=True)
    code = Column(String, nullable=False)

    def __repr__(self):
        return f"Code(id={self.id!r}, code={self.code!r})"


# Database model for global EA status.
class Status(Base):
    __tablename__ = "status"
    id = Column(Integer, primary_key=True)
    status = Column(Integer, nullable=False)
    iteration = Column(Integer, nullable=False)

    def __repr__(self):
        return f"Status(id={self.id!r}, status={self.status!r}, iteration={self.iteration!r})"


# Database model for generated code file paths.
class CodeFile(Base):
    __tablename__ = "code_file"
    id = Column(Integer, primary_key=True)
    file = Column(String, nullable=False)
    ckpt_dir = Column(String, nullable=False)
    save_dir = Column(String, nullable=False)
    log_dir = Column(String, nullable=False)

    def __repr__(self):
        return f"CodeFile(id={self.id!r}, file={self.file!r}, ckpt_dir={self.ckpt_dir!r}, " \
               f"save_dir={self.save_dir!r}, log_dir={self.log_dir!r})"


# Database model for result file paths.
class ResFile(Base):
    __tablename__ = "res_file"
    id = Column(Integer, primary_key=True)
    save_name = Column(String, nullable=False)
    save_dir = Column(String, nullable=False)

    def __repr__(self):
        return f"ResFile(id={self.id!r}, save_name={self.save_name!r}, save_dir={self.save_dir!r})"


# Database model for failed generated code.
class BadCode(Base):
    __tablename__ = "bad_code"
    id = Column(Integer, primary_key=True)
    train_log = Column(String, nullable=False)

    def __repr__(self):
        return f"BadCode(id={self.id!r}, train_log={self.train_log!r})"


# Database model for normalized expressed genes.
class GeneExpressed(Base):
    __tablename__ = "gene_expressed"
    id = Column(Integer, primary_key=True)
    gene_expressed = Column(String, nullable=False)

    def __repr__(self):
        return f"GeneExpressed(id={self.id!r}, gene_expressed={self.gene_expressed!r})"


# Database model for gene/code hash records.
class Hash(Base):
    __tablename__ = "hash"
    id = Column(Integer, primary_key=True)
    hash = Column(String, nullable=False)
    seq_info = Column(String, nullable=False)

    def __repr__(self):
        return f"Hash(id={self.id!r}, hash={self.hash!r})"


# Database model for pending training tasks.
class Task(Base):
    __tablename__ = "task"
    id = Column(Integer, primary_key=True)

    def __repr__(self):
        return f"Task(id={self.id!r})"


# Database model for claimed training tasks.
class TaskGet(Base):
    __tablename__ = "task_get"
    id = Column(Integer, primary_key=True)
    worker = Column(Integer, nullable=False)

    def __repr__(self):
        return f"TaskGet(id={self.id!r}, worker={self.worker!r})"


# Database model for registered workers.
class WorkerRegister(Base):
    __tablename__ = "worker_register"
    worker_id = Column(Integer, primary_key=True)
    lucky_string = Column(String, nullable=False)

    def __repr__(self):
        return f"WorkerRegister(worker_id={self.worker_id!r}, lucky_string={self.lucky_string!r})"


# Database model for completed training tasks.
class TaskEnd(Base):
    __tablename__ = "task_end"
    id = Column(Integer, primary_key=True)

    def __repr__(self):
        return f"TaskEnd(id={self.id!r})"


# Database model for differentiable-search tasks.
class DSRecord(Base):
    __tablename__ = "ds_record"
    ds_describe = Column(String, primary_key=True)
    iteration = Column(Integer, nullable=False)
    init_id = Column(Integer, nullable=False)
    init_gene = Column(String, nullable=False)
    init_score = Column(Float, nullable=False)
    cover_ids = Column(String, nullable=False)

    def __repr__(self):
        return f"DSRecord(ds_describe={self.ds_describe!r}, iteration={self.iteration!r}, " \
               f"init_id={self.init_id!r}, init_gene={self.init_gene!r}, init_score={self.init_score!r}," \
               f" cover_ids={self.cover_ids!r})"


# Database model for claimed differentiable-search tasks.
class DSTaskGet(Base):
    __tablename__ = "ds_task_get"
    ds_describe = Column(String, primary_key=True)
    worker = Column(Integer, nullable=False)

    def __repr__(self):
        return f"DSTaskGet(ds_describe={self.ds_describe!r}, worker={self.worker!r})"


# Database model for completed differentiable-search tasks.
class DSTaskEnd(Base):
    __tablename__ = "ds_task_end"
    ds_describe = Column(String, primary_key=True)

    def __repr__(self):
        return f"DSTaskEnd(ds_describe={self.ds_describe!r})"


# Database model for failed differentiable-search tasks.
class DSBadCode(Base):
    __tablename__ = "ds_bad_code"
    ds_describe = Column(String, primary_key=True)
    train_log = Column(String, nullable=False)

    def __repr__(self):
        return f"DSBadCode(ds_describe={self.ds_describe!r}, train_log={self.train_log!r})"


# Database model for differentiable-search results.
class DSResult(Base):
    __tablename__ = "ds_result"
    ds_describe = Column(String, primary_key=True)
    new_id = Column(Integer, nullable=False)
    new_score = Column(Float, nullable=False)

    def __repr__(self):
        return f"DSResult(ds_describe={self.ds_describe!r}, new_id={self.new_id!r}, " \
               f"new_score={self.new_score!r})"


# Create the database engine and initialize tables.
def create_db_engine():
    cfg = Config()

    engine = create_engine(
        cfg.db_conn_str,
        pool_recycle=cfg.DB_POOL_RECYCLE,
        pool_pre_ping=cfg.DB_POOL_PRE_PING,
        pool_use_lifo=cfg.DB_POOL_USE_LIFO,
        echo_pool=cfg.DB_ECHO_POOL,
        pool_size=cfg.DB_POOL_SIZE
    )

    mapper_registry.metadata.create_all(engine)
    return engine


# Read the latest EA status from the database.
def read_status(engine):
    with Session(engine) as session:
        row = session.query(Status).order_by(Status.id.desc()).first()
        if row is None:
            genes = [
'0-1,1,1,16,1,50,1-1,1,1,16,1,50,0-255',
'0-1,1,2,16,8,60,2-1,1,2,16,8,60,0-255',
'0-1,1,3,16,1,50,1-1,1,3,16,1,50,0-255',
'0-1,1,4,16,8,60,2-1,1,4,16,8,60,0-255',
'0-1,1,5,16,8,50,2-1,1,5,16,8,50,0-255',
'0-1,1,3,80,4,40,2-1,1,3,58,4,40,1-1,2,1,58,4,40,1-2,1,2,2-1,1,1,1,4,0,0-255',
'0-1,1,1,80,4,40,2-1,1,1,58,4,40,1-1,2,3,58,4,40,1-2,1,2,2-1,1,3,1,4,0,0-255',
'0-1,1,1,80,4,40,2-1,1,1,58,4,40,1-1,2,3,58,4,40,1-2,1,2,3-1,1,3,1,4,0,0-255',
'0-1,1,1,76,7,50,2-1,1,1,69,8,40,2-1,1,3,1,4,0,0-1,1,2,1,4,0,0-1,1,3,1,4,0,0-255',
'0-1,1,1,22,3,40,1-1,1,1,16,3,60,1-1,1,1,28,3,60,1-1,1,1,28,3,40,1-2,1,2,3-1,1,1,1,3,40,2-255',
'0-1,1,1,22,3,60,1-1,1,1,64,3,1,1-2,3,1,2-1,1,1,28,3,60,1-1,1,1,28,3,40,1-2,2,1,2-1,1,1,1,3,40,2-1,1,1,1,3,50,0-255',
'0-1,1,1,22,3,60,1-1,1,1,64,3,1,1-1,1,1,28,3,60,1-1,1,1,28,3,40,1-2,2,1,2-1,1,1,1,3,40,2-1,1,1,1,3,50,0-255',
'0-1,1,1,122,5,40,2-1,3,1,76,7,50,2-2,2,1,2-1,1,3,1,4,0,0-1,1,3,1,4,0,0-255',
'0-1,1,1,122,5,40,2-1,1,5,61,4,40,3-1,3,1,76,7,50,2-2,2,1,2-1,2,1,76,4,40,1-1,1,5,29,5,50,2-255',
'0-1,1,1,122,5,40,2-1,1,3,93,2,40,1-1,1,1,7,5,40,2-1,4,2,1,1,50,0-255',
'0-1,1,5,16,5,50,2-1,1,1,64,4,40,1-1,1,1,16,4,40,1-2,1,3,2-1,1,1,16,8,40,1-2,1,5,2-1,2,1,16,2,40,1-2,1,2,3-1,1,1,1,2,40,1,1-255',
'0-1,1,1,122,5,40,2-1,1,3,101,2,40,1-1,1,1,1,5,40,2-255'
]
            i = 1
            for gene in genes:
                r = Main(id=i, gene=gene)

                g = Generation(id=i, father=0, mother=0, iteration=0)
                session.add_all([r, g])
                i += 1
            s = Status(id=0, status=0, iteration=0)
            session.add(s)
            session.commit()

            status_dict = {
                'status': 0,
                'iteration': 0
            }
            return status_dict
        else:
            status_dict = {
                'status': row.status,
                'iteration': row.iteration,
            }
            session.commit()
            return status_dict

if __name__ == '__main__':
    engine = create_db_engine()

    print(read_status(engine))

    with Session(engine) as session:
        tables = Base.metadata.tables

        sorted_tables = reversed(Base.metadata.sorted_tables)

        for table in sorted_tables:
            session.execute(delete(table))

        session.commit()
    print("All database table data has been cleared")
