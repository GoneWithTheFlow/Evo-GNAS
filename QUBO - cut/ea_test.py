import init_ea
import ea
import ea_code_tf


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    engine = init_ea.create_db_engine()
    status = init_ea.read_status(engine)
    print("Status of ea: {}".format(status))
    print("\nStart EA Process")
    ea.ea_loop(engine, status)
