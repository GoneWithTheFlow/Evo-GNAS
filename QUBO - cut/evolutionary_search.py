import hashlib
import random
import time
from sqlalchemy.orm import Session


import architecture_codegen
import database
import job_recorder
from database import BadCode, Code, CodeFile, Generation, GeneExpressed, Hash, Length, Main, Number, ResFile, Runtime
from database import Status, Score, Acc, Task, TaskEnd, TaskGet, WorkerRegister
from differentiable_search import differentiable_search
from gnn_trainer import DATASET


import evaluator
from gene_graph import gene_gene_graph_with_info


# Read summary statistics for each evolutionary iteration.
def get_iteration_stats(engine, iteration: int):
    with Session(engine) as session:
        best_row = session.execute(
            "select score.id, score.score from score "
            "join generation on score.id=generation.id "
            "where generation.iteration={} "
            "order by score.score desc limit 1".format(iteration)
        ).first()

        main_count_row = session.execute(
            "select count(main.id) from main "
            "join generation on main.id=generation.id "
            "where generation.iteration<={}".format(iteration)
        ).first()
        main_count = main_count_row[0] if main_count_row and main_count_row[0] is not None else 0

        unique_count_row = session.execute(
            "select count(distinct hash.hash) from hash "
            "join generation on hash.id=generation.id "
            "where generation.iteration<={}".format(iteration)
        ).first()
        unique_count = unique_count_row[0] if unique_count_row and unique_count_row[0] is not None else 0

        train_count_row = session.execute(
            "select count(distinct runtime.id) from runtime "
            "join generation on runtime.id=generation.id "
            "where generation.iteration<={}".format(iteration)
        ).first()
        train_count = train_count_row[0] if train_count_row and train_count_row[0] is not None else 0

        runtime_sum_row = session.execute(
            "select sum(runtime.runtime) from runtime "
            "join generation on runtime.id=generation.id "
            "where generation.iteration={}".format(iteration)
        ).first()

        runtime_sum = runtime_sum_row[0] if runtime_sum_row and runtime_sum_row[0] is not None else 0.0

        if best_row:
            current_best_id = best_row[0]

            current_cut_row = session.execute(
                "select cut from acc where id={}".format(current_best_id)
            ).first()
            current_best_cut = current_cut_row[0] if current_cut_row and current_cut_row[0] is not None else best_row[1]

            prev_best_cut = float('-inf')
            if iteration > 0:
                prev_best_id_row = session.execute(
                    "select score.id from score "
                    "join generation on score.id=generation.id "
                    "where generation.iteration={} "
                    "order by score.score desc limit 1".format(iteration - 1)
                ).first()
                if prev_best_id_row:
                    prev_cut_row = session.execute(
                        "select cut from acc where id={}".format(prev_best_id_row[0])
                    ).first()
                    if prev_cut_row and prev_cut_row[0] is not None:
                        prev_best_cut = prev_cut_row[0]

            is_new_best = current_best_cut > prev_best_cut
            return {
                'iteration': iteration,
                'best_id': current_best_id,
                'best_cut': current_best_cut,
                'is_new_best': is_new_best,

                'total_count': unique_count,
                'main_count': main_count,
                'unique_count': unique_count,
                'train_count': train_count,
                'runtime_sum': runtime_sum
            }
    return None


# Select parent genes and generate the next evolutionary population.
def ea_select_generate(engine, iteration: int):
    iteration = iteration + 1
    print("iteration:", iteration)

    made = 0
    while made == 0:
        made = select_generate_gene(engine, iteration)

    with Session(engine) as session:
        s = session.query(database.Status).order_by(database.Status.id.desc()).first()

        id = s.id + 1

        r = database.Status(id=id, status=0, iteration=iteration)
        session.merge(r)
        session.commit()
    return iteration, made


# Compute a stable hash for a gene string.
def _gene_hash(gene: str) -> str:
    parts = [u.split(',') for u in gene.split('-')]
    fixed = fix_gene_params(parts)
    units = [','.join(u) for u in fixed][1:-1]
    seq, info = gene_gene_graph_with_info(units)
    return hashlib.sha1((seq + '|' + '-'.join(info)).encode('utf-8')).hexdigest()


# Select parent genes and create offspring genes.
def select_generate_gene(engine, iteration):
    with Session(engine) as session:
        cfg: database.Number = session.query(database.Number).order_by(database.Number.id.desc()).first()
        number, avg_length = cfg.number, cfg.avg_length

        row = session.execute(
            "select score.id from score "
            "join generation on score.id=generation.id "
            "where generation.iteration={} "
            "order by score.score desc limit 1".format(iteration - 1)
        ).first()
        if row is None:
            return 0
        elite_id = row[0]
        elite_gene = session.execute("select gene from main where id={}".format(elite_id)).first()[0]
        row = session.execute("select hash from hash where id={}".format(elite_id)).first()
        elite_h = row[0] if row else _gene_hash(elite_gene)

        want = max(2, number - 1)
        rows = session.execute(
            "select score.id, hash.hash from score "
            "join generation on score.id=generation.id "
            "left join hash on score.id=hash.id "
            "where generation.iteration={} "
            "order by score.score desc limit {}".format(iteration - 1, want * 4)
        ).all()

        parents, seen_parent_h = [], set()
        for pid, ph in rows:
            if ph is None:
                g = session.execute("select gene from main where id={}".format(pid)).first()[0]
                ph = _gene_hash(g)
            if ph in seen_parent_h:
                continue
            seen_parent_h.add(ph)
            parents.append(pid)
            if len(parents) >= want:
                break
        if len(parents) < 2:
            return 0

        need_ids = set(parents + [elite_id])
        parent_gene = {pid: session.execute("select gene from main where id={}".format(pid)).first()[0]
                       for pid in need_ids}

        child_h = {elite_h}
        made = 0
        pos = list(range(len(parents)))

        row = session.execute("select id from main order by id desc limit 1").first()
        next_id = (row[0] if row else 0) + 1

        max_tries = max(2000, (number - 1) * 200)
        for _ in range(max_tries):
            if made >= number - 1:
                break

            i, j = random.sample(pos, 2)
            father, mother = parents[min(i, j)], parents[max(i, j)]

            gene = crossover_gene(parent_gene[father], parent_gene[mother], iteration)
            gene = mutant_gene(gene, iteration, avg_length)

            h = _gene_hash(gene)
            if h in child_h:
                continue

            session.merge(Generation(id=next_id, father=father, mother=mother, iteration=iteration))
            session.merge(Main(id=next_id, gene=gene))
            child_h.add(h)

            next_id += 1
            made += 1

        session.merge(Generation(id=next_id, father=elite_id, mother=elite_id, iteration=iteration))
        session.merge(Main(id=next_id, gene=elite_gene))

        session.commit()
        return made


# Generate a random valid gene for the initial population.
def generate_gene(engine, couple, iteration, avg_length):
    with Session(engine) as session:
        row_father = session.execute("select * from main where id={}".format(couple[0],)).first()
        row_mother = session.execute("select * from main where id={}".format(couple[1],)).first()
        session.commit()

    gene_f = row_father[1]
    gene_m = row_mother[1]

    gene = crossover_gene(gene_f, gene_m, iteration)

    gene = mutant_gene(gene, iteration, avg_length)

    with Session(engine) as session:
        row = session.execute("select id from main order by id desc limit 1").first()
        id = row[0] + 1

        row1 = Generation(id=id,
                          father=couple[0],
                          mother=couple[1],
                          iteration=iteration
                          )
        session.merge(row1)
        session.commit()

        row2 = Main(id=id,
                    gene=gene
                    )
        session.merge(row2)
        session.commit()


# Create a child gene by crossing over two parent genes.
def crossover_gene(gene_a: str, gene_b: str, iteration: int):
    gene_a_c = gene_a
    gene_b_c = gene_b

    gene_a_units = gene_a.split('-')[1:-1]
    gene_b_units = gene_b.split('-')[1:-1]
    gene_a_len = len(gene_a_units)
    gene_b_len = len(gene_b_units)

    new_gene_a_units = []
    new_gene_b_units = []
    for i in range(max(gene_a_len, gene_b_len)):
        if i >= gene_a_len:
            temp_a = None
        else:
            temp_a = gene_a_units[i]
        if i >= gene_b_len:
            temp_b = None
        else:
            temp_b = gene_b_units[i]

        if random.randint(0, 1) == 0:
            if temp_a is not None:
                new_gene_a_units.append(temp_a)
            if temp_b is not None:
                new_gene_b_units.append(temp_b)
        else:
            if temp_a is not None:
                new_gene_b_units.append(temp_a)
            if temp_b is not None:
                new_gene_a_units.append(temp_b)

    new_gene_a_units = ['0'] + new_gene_a_units + ['255']
    new_gene_b_units = ['0'] + new_gene_b_units + ['255']

    gene_a = '-'.join(new_gene_a_units)
    gene_b = '-'.join(new_gene_b_units)

    if iteration < 1:
        iteration = 1

    p_crossover = 1

    which_gene = random.randint(0, 1)

    if random.random() < p_crossover:
        if which_gene == 0:
            gene = gene_a
        else:
            gene = gene_b
    else:
        if which_gene == 0:
            gene = gene_a_c
        else:
            gene = gene_b_c
    return gene


# Mutate a gene by changing structure or operation parameters.
def mutant_gene(gene: str, iteration: int, avg_length: int):
    if iteration < 1:
        iteration = 1
    if avg_length < 1:
        avg_length = 1

    if random.random() >= 1 / (iteration ** (1 / float(avg_length))):
        return gene

    gene_units = gene.split('-')

    mutant_type = random.randint(0, 2)

    if len(gene_units) == 2:
        mutant_type = 1

    if mutant_type == 0:
        if len(gene_units) > 3:
            unit_index = random.randint(1, len(gene_units) - 2)

            gene_units = gene_units[:unit_index] + gene_units[unit_index + 1:]
        gene = '-'.join(gene_units)
        return gene
    elif mutant_type == 1:
        unit_index = random.randint(1, len(gene_units) - 1)

        unit_type = random.randint(0, 9)

        if unit_type == 1:
            unit_params = ['1',
                           str(random.randint(1, unit_index)),
                           '1',
                           str(random.randint(16, 128)),
                           str(random.randint(4, 8)),
                           str(random.choice([30,40,50])),
                           str(random.randint(0, 3))
                           ]
        elif unit_type == 2:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '2',
                str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30,40,50])),
                str(random.randint(0, 3))
            ]
        elif unit_type == 3:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '3',
                str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30,40,50])),
                str(random.randint(0, 3))
            ]
        elif unit_type == 4:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '4',
                str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30,40,50])),
                str(random.randint(0, 3))
            ]
        elif unit_type == 5:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '5',
                str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30,40,50])),
                str(random.randint(0, 3))
            ]
        elif unit_type == 6:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '6',
                str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30, 40, 50])),
                str(random.randint(0, 3))
            ]
        elif unit_type == 7:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '7',
                str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30, 40, 50])),
                str(random.randint(0, 3))
            ]
        elif unit_type == 8:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '8',
                str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30, 40, 50])),
                str(random.randint(0, 3))
            ]
        elif unit_type == 9:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '9',
                str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30, 40, 50])),
                str(random.randint(0, 3))
            ]
        else:
            unit_params = ['2',
                           str(random.randint(1, unit_index)),
                           str(random.randint(1, unit_index)),
                           str(random.randint(2, 3))
                           ]

        gene_units.insert(unit_index, ','.join(unit_params))
        gene = '-'.join(gene_units)
        return gene

    unit_index = random.randint(1, len(gene_units) - 2)

    unit_params = gene_units[unit_index].split(',')

    unit_input_num = int(unit_params[0])

    unit_type = int(unit_params[unit_input_num + 1])

    if unit_input_num == 2:
        part_mutant = random.randint(0, 1)
    else:
        part_mutant = random.randint(0, 2)

    if part_mutant == 0:
        input_mutant = random.randint(1, unit_input_num)
        base = int(unit_params[input_mutant])

        unit_params[input_mutant] = str(random.randint(base - int(base / 2), int((base + 1) * 1.5)))
    elif part_mutant == 1:
        type_mutant = random.randint(0, 9)
        if type_mutant == 1:
            unit_params = ['1',
                           str(random.randint(1, unit_index)),
                           '1',
                           str(random.randint(16, 128)),
                           str(random.randint(4, 8)),
                           str(random.choice([30,40,50])),
                           str(random.randint(0, 3))
                           ]
        elif type_mutant == 2:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '2',
                str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30,40,50])),
                str(random.randint(0, 3))
            ]
        elif type_mutant == 3:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '3',
                str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30,40,50])),
                str(random.randint(0, 3))
            ]
        elif type_mutant == 4:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '4',
                str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30,40,50])),
                str(random.randint(0, 3))
            ]
        elif type_mutant == 5:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '5',
                str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30,40,50])),
                str(random.randint(0, 3))
            ]
        elif type_mutant == 6:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '6',
                str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30, 40, 50])),
                str(random.randint(0, 3))
            ]
        elif type_mutant == 7:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '7',
                str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30, 40, 50])),
                str(random.randint(0, 3))
            ]
        elif type_mutant == 8:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '8',
                unit_params[3] if unit_input_num == 1 else str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30, 40, 50])),
                str(random.randint(0, 3))
            ]
        elif type_mutant == 9:
            unit_params = [
                '1',
                str(random.randint(1, unit_index)),
                '9',
                str(random.randint(16, 128)),
                str(random.randint(4, 8)),
                str(random.choice([30, 40, 50])),
                str(random.randint(0, 3))
            ]
        else:
            if unit_input_num == 2:
                unit_params = ['2',
                               unit_params[1],
                               unit_params[2],
                               str(random.randint(2, 3))
                               ]
            else:
                if unit_params[1] == '1':
                    second_input = str(random.randint(2, min(8, len(gene_units) - 2)))
                    unit_params = ['2', '1', second_input, str(random.randint(2, 3))]
                else:
                    unit_params = ['2',
                                   '1',
                                   unit_params[1],
                                   str(random.randint(2, 3))
                                   ]
    elif part_mutant == 2:
        if unit_type in [1, 3, 9]:
            mutant_pos = random.choice([3,5,6])
            if mutant_pos == 3:
                base = int(unit_params[3])

                lower = max(base - int(base / 2), 16)
                upper = min(int((base + 1) * 1.5), 128)

                if lower > upper:
                    upper = lower

                unit_params[3] = str(random.randint(lower, upper))
            elif mutant_pos == 5:
                unit_params[5] = str(random.choice([30,40,50]))
            elif mutant_pos == 6:
                unit_params[6] = str(random.randint(0, 3))
        elif unit_type in [2, 4, 5, 6, 7]:
            mutant_pos = random.randint(3, 6)
            if mutant_pos == 3:
                base = int(unit_params[3])

                lower = max(base - int(base / 2), 16)
                upper = min(int((base + 1) * 1.5), 128)

                if lower > upper:
                    upper = lower

                unit_params[3] = str(random.randint(lower, upper))
            elif mutant_pos == 4:
                unit_params[4] = str(random.randint(4, 8))
            elif mutant_pos == 5:
                unit_params[5] = str(random.choice([30,40,50]))
            elif mutant_pos == 6:
                unit_params[6] = str(random.randint(0, 3))
        elif unit_type == 8:
            mutant_pos = random.choice([5, 6])
            if mutant_pos == 5:
                unit_params[5] = str(random.choice([30, 40, 50]))
            elif mutant_pos == 6:
                unit_params[6] = str(random.randint(0, 3))

    gene_units[unit_index] = ','.join(unit_params)
    gene = '-'.join(gene_units)
    return gene


# Create training tasks and score the generated population.
def ea_train_score(engine, iteration: int):
    with Session(engine) as session:
        for row in session.execute("select * from main where id not in (select id from code)").all():
            express_gene(engine, row[0], row[1])

        for row in session.execute("select * from code where id not in (select id from score) "
                                   "and id not in (select id from bad_code)").all():
            train_score_code(engine, row[0], row[1])
        session.commit()

    wait_train_workers(engine)

    with Session(engine) as session:
        row = session.execute("select * from status order by id desc limit 1").first()
        id = row[0] + 1

        row = Status(id=id, status=1, iteration=iteration)
        session.merge(row)
        session.commit()
    return iteration


# Wait until all GNN training workers finish their tasks.
def wait_train_workers(engine):
    with Session(engine) as session:
        row = session.execute("select * from task where id not in (select id from task_end)").first()

        while row is not None:
            time.sleep(60)
            row = session.execute("select * from task where id not in (select id from task_end)").first()
        session.commit()


# Normalize a raw gene, store its graph hash, and generate model code.
def express_gene(engine, id: int, gene: str):
    gene_units = gene.split('-')

    gene_units_params = [unit.split(',') for unit in gene_units]

    gene_units_params_fixed = fix_gene_params(gene_units_params)

    with Session(engine) as session:
        row = GeneExpressed(id=id, gene_expressed='-'.join(
            [','.join(unit_params) for unit_params in gene_units_params_fixed]))
        session.merge(row)
        session.commit()

    gene_units = [','.join(unit_params) for unit_params in gene_units_params_fixed][1:-1]

    seq, info = gene_gene_graph_with_info(gene_units)
    info_s = '-'.join(info)

    h = hashlib.sha1((seq + '|' + info_s).encode(encoding='utf-8')).hexdigest()

    with Session(engine) as session:
        row = Hash(id=id,
                   hash=h,
                   seq_info=seq + '|' + info_s
                   )
        session.merge(row)
        session.commit()

    code, gene_code_length = architecture_codegen.gene_to_code(gene_units_params_fixed)
    gene_length = len(gene_units_params) - 2
    with Session(engine) as session:
        row1 = Code(id=id, code=code)
        session.merge(row1)
        session.commit()

        row2 = Length(id=id, length=gene_length, gene_code_length=gene_code_length)
        session.merge(row2)
        session.commit()


# Repeatedly repair gene parameters until input bounds and shapes are valid.
def fix_gene_params(gene_units_params):
    gene_before = '-'.join([','.join(unit_params) for unit_params in gene_units_params])

    gene_units_params = fix_gene_input_bounds(gene_units_params)

    gene_units_params = fix_gene_input_shape(gene_units_params)

    gene_after = '-'.join([','.join(unit_params) for unit_params in gene_units_params])

    while gene_before != gene_after:
        gene_before = gene_after

        gene_units_params= fix_gene_input_bounds(gene_units_params)

        gene_units_params = fix_gene_input_shape(gene_units_params)

        gene_after = '-'.join([','.join(unit_params) for unit_params in gene_units_params])
    return gene_units_params


# Clamp gene input offsets so every unit only references previous units.
def fix_gene_input_bounds(gene_units_params):
    pos = 0
    for i in range(len(gene_units_params)):
        if gene_units_params[i][0] == '0':
            continue
        if gene_units_params[i][0] == '255':
            break
        pos = pos + 1
        for j in range(int(gene_units_params[i][0])):
            if int(gene_units_params[i][j + 1]) > pos:
                gene_units_params[i][j + 1] = str(pos)

    return gene_units_params


# Insert or adjust units so gene outputs have the required feature shape.
def fix_gene_input_shape(gene_units_params):
    dim_record = []
    unit_insert = []
    for i in range(len(gene_units_params)):
        if gene_units_params[i][0] == '0':
            dim_record.append((i, DATASET.in_channels))
            continue
        if gene_units_params[i][0] == '255':
            if dim_record[-1][1] != DATASET.out_channels:
                last_unit_idx = dim_record[-1][0]
                last_unit_input_num = int(gene_units_params[last_unit_idx][0])
                last_unit_type = int(gene_units_params[last_unit_idx][last_unit_input_num + 1])
                if last_unit_input_num == 1:
                    if last_unit_type == 8:
                        unit_insert.append((i, ['1', '1', '9', str(DATASET.out_channels), '0', '40', '0']))
                    else:
                        gene_units_params[last_unit_idx][3] = str(DATASET.out_channels)
                        if last_unit_type == 1:
                            pass
                        elif last_unit_type == 2:
                            gene_units_params[last_unit_idx][4] = '1'
                        elif last_unit_type == 4:
                            gene_units_params[last_unit_idx][4] = '1'
                else:
                    unit_insert.append((i, ['1', '1', '1', str(DATASET.out_channels), '4', '40', '0']))
            break

        input_num = int(gene_units_params[i][0])
        type = int(gene_units_params[i][input_num + 1])
        if input_num == 1:
            try:
                offset = int(gene_units_params[i][1])
            except Exception:
                offset = 1
            offset = max(1, min(offset, len(dim_record)))
            in_dim = dim_record[-offset][1]

            if type == 8:
                gene_units_params[i][3] = str(in_dim)
                dim = in_dim
            else:
                dim = int(gene_units_params[i][3])

            dim_record.append((i, dim))
        elif input_num == 2:
            offset_a = int(gene_units_params[i][1])
            offset_b = int(gene_units_params[i][2])
            channel_a = dim_record[-offset_a][1]
            channel_b = dim_record[-offset_b][1]
            if channel_a == channel_b:
                dim_record.append((i, channel_a))
            else:
                unit_a_idx = dim_record[-offset_a][0]
                unit_b_idx = dim_record[-offset_b][0]

                if (channel_a > channel_b and gene_units_params[unit_a_idx][0] != '0') \
                        or gene_units_params[unit_b_idx][0] == '0':
                    type_a = int(gene_units_params[unit_a_idx][int(gene_units_params[unit_a_idx][0]) + 1])
                    if gene_units_params[unit_a_idx][0]==1:
                        gene_units_params[unit_a_idx][3] = str(channel_b)

                        dim_record[-offset_a] = (unit_a_idx, channel_b)
                        dim_record.append((i, channel_b))
                    else:
                        unit_insert.append((i, ['1', gene_units_params[i][1], '1', str(channel_b), '4', '40', '1']))

                        gene_units_params[i][1] = '1'
                        gene_units_params[i][2] = str(int(gene_units_params[i][2]) + 1)
                        dim_record.append((i, channel_b))
                else:
                    type_b = int(gene_units_params[unit_b_idx][int(gene_units_params[unit_b_idx][0]) + 1])
                    if gene_units_params[unit_b_idx][0]==1:
                        gene_units_params[unit_b_idx][3] = str(channel_a)

                        dim_record[-offset_b] = (unit_b_idx, channel_a)
                        dim_record.append((i, channel_a))
                    else:
                        unit_insert.append((i, ['1', gene_units_params[i][2], '1', str(channel_a), '4', '40', '1']))

                        gene_units_params[i][2] = '1'
                        gene_units_params[i][1] = str(int(gene_units_params[i][1]) + 1)
                        dim_record.append((i, channel_a))
    gene_units_params = gene_insert_units(gene_units_params, unit_insert)
    return gene_units_params


# Find a nearby divisor used to split feature dimensions safely.
def find_near_divide(k, s, t):
    if k < 1:
        return 1
    if s != t:
        if s > t:
            p = t
            t = s
            s = p
        r = t % s
        while r > 0:
            t = s
            s = r
            r = t % s
    if k > s:
        return s
    if s % k == 0:
        return k
    d = 1
    while d < s / 2 + 1:
        if s % (k - d) == 0:
            return k - d
        if s % (k + d) == 0:
            return k + d
        d = d + 1
    return 1


# Insert repair units into a gene while preserving unit order.
def gene_insert_units(gene_units_params, unit_insert):
    insert_offset = 0
    for insert in unit_insert:
        insert_pos = insert[0] + insert_offset
        gene_units_params.insert(insert_pos, insert[1])
        pos = 1
        next_input_num = int(gene_units_params[insert_pos + 1][0])
        if next_input_num == 0 or next_input_num == 255:
            continue
        if gene_units_params[insert_pos + 1][next_input_num + 1] == '5':
            pos = pos + int(gene_units_params[insert_pos + 1][3]) - 1
        for i in range(insert_pos + 2, len(gene_units_params) - 1):
            input_num = int(gene_units_params[i][0])
            for j in range(input_num):
                offset = int(gene_units_params[i][j + 1])
                if offset > pos:
                    gene_units_params[i][j + 1] = str(offset + 1)
            if gene_units_params[i][input_num + 1] == '5':
                pos = pos + int(gene_units_params[i][3])
            else:
                pos = pos + 1
        insert_offset = insert_offset + 1
    return gene_units_params


# Train generated code and store the resulting evaluation score.
def train_score_code(engine, id: int, code: str):
    with Session(engine) as session:
        row = session.execute("select * from code_file where id={}".format(id,)).first()
        if row is None:
            code_filename, code_name = architecture_codegen.code_to_file(id, code)

            ckpt_dir = './ckpt/' + code_name
            save_dir = './save/' + code_name
            log_dir = './logs/' + code_name

            row = CodeFile(id=id, file=code_filename, ckpt_dir=ckpt_dir, save_dir=save_dir, log_dir=log_dir)
            session.merge(row)
            session.commit()

        sql = 'select accuracy,cut from acc where id in (select id from hash where hash=(select hash from hash where id={})) ' \
              'order by accuracy desc limit 1'.format(id)
        row = session.execute(sql).first()

        if row is not None:
            accuracy = row[0]
            cut = row[1]

            row = session.execute("select * from acc where id={}".format(id,)).first()
            if row is None:
                row = Acc(id=id, accuracy=accuracy,cut=cut)
                session.merge(row)
                session.commit()
                return

        if session.query(Task).where(Task.id==id).first() is None:
            row = Task(id=id)
            session.merge(row)
            session.commit()

        job_recorder.request_evaluation([id])


# Run differentiable search after obtaining the score.
def ea_number(engine, iteration: int):
    with Session(engine) as session:
        rows = session.execute("select acc.id from generation inner join acc on "
                               "generation.id=acc.id where iteration={}".format(iteration)).all()
    id_list = []
    for row in rows:
        id_list.append(row[0])

    with Session(engine) as session:
        for gene_id in id_list:
            row = session.execute(
                "select accuracy from acc where id={}".format(gene_id)
            ).first()
            if row:
                acc = row[0]

                score_row = Score(id=gene_id, score=acc)
                session.merge(score_row)
        session.commit()

    differentiable_search(engine, iteration)

    with Session(engine) as session:
        row = session.execute("select AVG(length) from length").first()
        avg_length = int(row[0])

        number = avg_length * 2

        row = session.execute("select id from number order by id desc limit 1").first()
        if row is None:
            id = 0
        else:
            id = row[0] + 1

        row = Number(id=id, number=number, avg_length=avg_length)
        session.merge(row)
        session.commit()

        row = session.execute("select * from status order by id desc limit 1").first()
        id = row[0] + 1
        row = Status(id=id, status=2, iteration=iteration)
        session.merge(row)
        session.commit()
    return iteration


# Run the main evolutionary search loop.
def ea_loop(engine, status_dict):
    iteration = status_dict['iteration']
    print("iteration of ea: {}".format(iteration))

    log_file = "evolution.txt"

    if status_dict['status'] == 0:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Evolutionary Algorithm (EA) Training Statistics\n")
            f.write("="*80 + "\n\n")

        iteration = ea_train_score(engine, iteration)

        iteration = ea_number(engine, iteration)

        stats = get_iteration_stats(engine, iteration)
        if stats:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[Initialization] Iteration {stats['iteration']}: "
                       f"Current best ID={stats['best_id']}, Max cut={stats['best_cut']:.1f}, "
                       f"Main records={stats['main_count']}, "
                       f"Actual trainings={stats['train_count']}, Runtime={stats['runtime_sum']:.2f}s\n")
    elif status_dict['status'] == 1:
        iteration = ea_number(engine, iteration)

    max_iterations = 20
    while iteration < max_iterations:
        iteration, newly_generated = ea_select_generate(engine, iteration)

        iteration = ea_train_score(engine, iteration)

        iteration = ea_number(engine, iteration)

        stats = get_iteration_stats(engine, iteration)
        if stats:
            marker = " * [New best]" if stats['is_new_best'] else ""
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Iteration {stats['iteration']:>2}: New generated={newly_generated:>3}, "
                       f"Main records={stats['main_count']:>4}, "
                       f"Train={stats['train_count']:>4}, "
                       f"Best ID={stats['best_id']:>5}, Max cut={stats['best_cut']:.1f}, "
                       f"Runtime={stats['runtime_sum']:>8.2f}s{marker}\n")

    print("Evolution completed!")

    with Session(engine) as session:
        global_best = session.execute(
            "select score.id, score.score from score order by score.score desc limit 1"
        ).first()

        total_runtime_row = session.execute(
            "select sum(runtime) from runtime"
        ).first()
        total_runtime = total_runtime_row[0] if total_runtime_row and total_runtime_row[0] is not None else 0.0

        if global_best:
            best_id = global_best[0]

            cut_row = session.execute(
                f"select cut from acc where id={best_id}"
            ).first()
            cut_value = cut_row[0] if cut_row else None

            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
                f.write("Final Result\n")
                f.write("="*80 + "\n")
                f.write(f"Global best architecture ID: {best_id}\n")
                if cut_value is not None:
                    f.write(f"Max cut value: {cut_value:.1f}\n")
                f.write(f"Total runtime: {total_runtime:.2f}s ({total_runtime/3600:.2f}h)\n")
                f.write("="*80 + "\n")

            print(f"Global best architecture ID={best_id}, Max cut={cut_value}")
            print(f"Total runtime: {total_runtime:.2f}s ({total_runtime/3600:.2f}h)")

    print(f"Statistics saved to {log_file}")
