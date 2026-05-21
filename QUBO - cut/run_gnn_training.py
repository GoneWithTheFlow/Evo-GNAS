import time
import argparse
import os
import subprocess


import database
from sqlalchemy.orm import Session


parser = argparse.ArgumentParser(description="A worker for evolutionary-search GNN training.")


parser.add_argument('-n', '--name', type=str, help="Unique name of worker.")
parser.add_argument('-g', '--gpu', type=int, help="Index of GPU.")


args = parser.parse_args()


# Load training task metadata.
def get_train_task_info(engine, id: int):
    with Session(engine) as session:
        row = session.execute("select * from code_file where id={}".format(id)).first()
        session.commit()

    ckpt_dir = row[2]
    save_dir = row[3]
    log_dir = row[4]
    return ckpt_dir, save_dir, log_dir


# Check whether a training or search result already exists.
def try_score_info(engine, id: int):
    with Session(engine) as session:
        row = session.execute(f"SELECT * FROM acc WHERE id={id}").first()
        session.commit()

    if row is None:
        return 404
    else:
        return 200


# Register a worker process in the task database.
def worker_register(engine, lucky_string: str):
    if lucky_string.count('%') > 0 or lucky_string.count('_') > 0:
        return -1

    with Session(engine) as session:
        row = session.execute("select worker_id from worker_register "
                              "where lucky_string like '{}'".format(lucky_string)).first()

        if row is None:
            row = session.execute("select count(*) from worker_register").first()
            count = row[0]
            print("New worker id: ", count, " Name: ", lucky_string)

            session.execute("insert into worker_register values ({}, '{}')".format(count, lucky_string))
            session.commit()
            return count
        else:
            session.commit()
            return row[0]


# Claim the next unfinished task from the database.
def find_task(engine, worker_id: int):
    with Session(engine) as session:
        while True:
            try:
                row = session.execute("select id from task where id not in (select id from task_get)").first()
                if row is None:
                    time.sleep(60)
                    continue

                id = row[0]

                session.execute("insert into task_get values ({}, {})".format(id, worker_id))
                session.commit()

                time.sleep(10)
                row = session.execute("select id from task_get where id={} and worker={}".format(id, worker_id)).first()
                if row is None:
                    continue
                else:
                    return id
            except:
                time.sleep(10)
                continue
        session.commit()


# Check whether the current task should continue running.
def check_task_continue(engine, worker_id: int):
    with Session(engine) as session:
        row = session.execute("select id from task_get where worker={} and "
                              "id not in (select id from task_end)".format(worker_id)).first()
        session.commit()
    if row is None:
        return -1
    else:
        return row[0]


# Mark a worker task as finished in the database.
def finish_task(engine, id: int):
    with Session(engine) as session:
        row = session.execute("select id from task_end where id={}".format(id)).first()

        if row is None:
            session.execute("insert into task_end values ({})".format(id))
        session.commit()


# Record failed generated code and its training log.
def log_bad_code(engine, id: int):
    with Session(engine) as session:
        session.execute("insert into bad_code values ({}, '{}')".format(id, './train_outputs/{}.out'.format(id)))
        session.commit()


# Execute one claimed task from code generation through scoring.
def do_task(engine, id: int, gpu: int, lucky_string: str):
    ckpt_dir, save_dir, log_dir = get_train_task_info(engine, id)

    if try_score_info(engine, id) == 200:
        print("task ", id, " is done, and submitting.")
        finish_task(engine, id)
        print("task ", id, " is done, and submitted.")
        return

    train_script = '''import gnn_trainer
import torch
import os
from database import create_db_engine
from sqlalchemy.orm import Session
import models.model_{} as model_gen


id = {}
save_dir = "{}"
os.makedirs(save_dir, exist_ok=True)


final_model_dirname, best_cut, acc, runtime = gnn_trainer.train(
    model_class=model_gen.GEN_GNN_MODEL,
    save_dir=save_dir
)


engine = create_db_engine()
with Session(engine) as session:
    session.execute("insert into res_file values ("+str(id)+", '"+str(final_model_dirname)+"', '"+str(save_dir)+"')")
    session.execute("insert into acc (id, accuracy, cut) values ("+str(id)+", "+str(acc)+", "+str(best_cut)+")")
    session.execute("insert into runtime (id, runtime) values (" + str(id) + ", " + str(runtime) + ")")
    session.commit()
engine.dispose()
'''.format(id, id, save_dir)

    script_path = './train_script_gen_' + lucky_string + '.py'
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(train_script)

    if not os.path.exists("./train_outputs/"):
        os.mkdir("./train_outputs/")

    with open('./train_outputs/{}.out'.format(id), 'a', encoding='utf-8') as f:
        return_code = subprocess.call(['python', '-u', script_path], stdout=f, stderr=f)

    if return_code != 0:
        print("ERROR: Train process of id={} model is failed, return code={}, please check output log at ./train_outputs/{}.out".format(
            id, return_code, id))
        log_bad_code(engine, id)

    print("task ", id, " is done, and submitting.")
    finish_task(engine, id)
    print("task ", id, " is done, and submitted.")


# Run the worker loop until all available tasks are processed.
def workflow(lucky_string: str, gpu: int):
    engine = database.create_db_engine()

    worker_id = worker_register(engine, lucky_string)
    while True:
        id = check_task_continue(engine, worker_id)
        while id != -1:
            print("find undone task ", id)
            do_task(engine, id, gpu, lucky_string)
            id = check_task_continue(engine, worker_id)

        id = find_task(engine, worker_id)
        do_task(engine, id, gpu, lucky_string)


def main():
    workflow(args.name, args.gpu)


# Local worker smoke-test entry point.
def test():
    engine = database.create_db_engine()

    test_id = 1
    gpu_idx = 0
    worker_name = "test_worker"

    do_task(engine, test_id, gpu_idx, worker_name)

if __name__ == '__main__':
    main()
