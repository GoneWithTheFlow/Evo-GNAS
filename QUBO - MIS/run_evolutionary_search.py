import database
import evolutionary_search
import architecture_codegen


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    engine = database.create_db_engine()
    status = database.read_status(engine)
    print("Status of ea: {}".format(status))
    print("\nStart EA Process")
    evolutionary_search.ea_loop(engine, status)
