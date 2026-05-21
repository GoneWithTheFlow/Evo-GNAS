import time
from typing import List, Dict


from sqlalchemy.orm import Session


from ea_code_tf import optim_gene_code
from init_ea import DSRecord


# Run differentiable search tasks for promising architectures.
def differentiable_search(engine, iteration: int):
    ds_dict = get_scored_individual_space(engine, iteration)
    ds_dict = get_unsearched_space(engine, ds_dict)
    init_ds_tasks(engine, ds_dict, iteration)
    wait_ds_workers(engine)


# Wait until all differentiable-search workers finish their tasks.
def wait_ds_workers(engine):
    with Session(engine) as session:
        row = session.execute("select * from ds_record "
                              "where ds_describe not in (select ds_describe from ds_task_end)").first()

        while row is not None:
            time.sleep(60)

            row = session.execute("select * from ds_record "
                                  "where ds_describe not in (select ds_describe from ds_task_end)").first()
        session.commit()


# Initialize differentiable-search tasks from evolutionary candidates.
def init_ds_tasks(engine, ds_dict: Dict[str, List[tuple]], iteration):
    with Session(engine) as session:
        for key in list(ds_dict.keys()):
            value = ds_dict[key]
            if len(value) == 1:
                item = value[0]

                task = {"ds_describe": key,
                        "init_id": item[0],
                        "init_gene": item[1],
                        "init_score": item[2],
                        "cover_ids": str(item[0])
                        }
            else:
                item = value[0]
                for i in range(1, len(value)):
                    if value[i][2] > item[2]:
                        item = value[i]

                cover_ids = ",".join([str(i[0]) for i in value])

                task = {"ds_describe": key,
                        "init_id": item[0],
                        "init_gene": item[1],
                        "init_score": item[2],
                        "cover_ids": cover_ids
                        }

            row = DSRecord(ds_describe=task["ds_describe"], iteration=iteration, init_id=task["init_id"],
                           init_gene=task["init_gene"], init_score=task["init_score"], cover_ids=task["cover_ids"])
            session.merge(row)
        session.commit()


# Select candidate genes that have not yet been explored by differentiable search.
def get_unsearched_space(engine, ds_dict: Dict[str, List[tuple]]):
    with Session(engine) as session:
        keys = ",".join(["'" + i + "'" for i in list(ds_dict.keys())])
        if not keys:
            rows = []
        else:
            rows = session.execute("select ds_describe from ds_record where ds_describe in ({})".format(keys)).all()
        session.commit()

        for row in rows:
            ds_dict.pop(row[0])

    return ds_dict


# Collect scored individual genes for differentiable-search initialization.
def get_scored_individual_space(engine, iteration: int):
    with Session(engine) as session:
        rows = session.execute("select gene_expressed.id, gene_expressed.gene_expressed, score.score from gene_expressed "
                               "inner join score on gene_expressed.id=score.id "
                               "inner join generation on gene_expressed.id=generation.id "
                               "where generation.iteration={}".format(iteration)).all()

        session.commit()
        ds_dict = {}
        for row in rows:
            id = row[0]
            gene = row[1]
            score = row[2]

            gene = get_reduct_gene(gene)

            ds_describe = get_dspace_describe(gene)

            if ds_describe in ds_dict:
                ds_dict[ds_describe] = ds_dict[ds_describe] + [(id, gene, score)]
            else:
                ds_dict[ds_describe] = [(id, gene, score)]
    return ds_dict


# Build the differentiable-search space description for a gene.
def get_dspace_describe(gene: str):
    gene_units = get_units_by_gene(gene)
    params_list = get_params_list_by_units(gene_units)
    for i in range(len(params_list)):
        if params_list[i][0] == 1:
            params_list[i] = params_list[i][:2]
        elif params_list[i][0] == 2:
            params_list[i] = params_list[i][:4]

    ds_describe = "-".join([",".join([str(i) for i in params]) for params in params_list])
    return ds_describe


# Reduce a gene by replacing searchable choices with placeholders.
def get_reduct_gene(gene: str):
    gene = get_optim_gene(gene)

    adj_matrix, params = get_adjacency_matrix_with_params(gene)
    adj_matrix, params = clean_matrix_with_params(adj_matrix, params)
    sort_matrix_with_params(adj_matrix, params)
    adj_matrix, params = clean_matrix_with_params(adj_matrix, params)
    gene = get_gene_by_matrix_with_params(adj_matrix, params)
    return gene


# Choose the best gene variant from differentiable-search results.
def get_optim_gene(gene):
    units = get_units_by_gene(gene)
    params_list = get_params_list_by_units(units)
    optim_params_list = params_list_s2i(optim_gene_code(params_list_i2s(params_list)))
    nodes = [[i, optim_params_list[i]] for i in range(len(optim_params_list))]
    idx_map = {}
    new_nodes = []
    for i in range(len(nodes)):
        if nodes[i][1][0] == 255:
            new_nodes.append(nodes[i])
            continue
        if nodes[i][1][0] == 0:
            new_nodes.append(nodes[i])
            idx_map[nodes[i][0]] = len(new_nodes) - 1
            continue
        if nodes[i][1][0] > 0:
            new_nodes.append(nodes[i])
            self_idx = len(new_nodes) - 1
            idx_map[nodes[i][0]] = self_idx
            for j in range(nodes[i][1][0]):
                idx = nodes[i][0] - nodes[i][1][1 + j]
                new_nodes[-1][1][1 + j] = self_idx - idx_map[idx]
    new_nodes = clean_same_add(new_nodes)
    return get_gene_by_nodes(new_nodes)


# Split a gene string into gene units.
def get_units_by_gene(gene):
    return gene.split("-")


# Convert gene units into parameter lists.
def get_params_list_by_units(units):
    def get_params_by_unit(unit):
        return [int(i) for i in unit.split(",")]

    return [get_params_by_unit(unit) for unit in units]


# Convert parameter lists from integers to strings.
def params_list_i2s(params_list):
    return [[str(param) for param in params] for params in params_list]


# Convert parameter lists from strings to integers.
def params_list_s2i(params_list):
    return [[int(param) for param in params] for params in params_list]


# Remove duplicate add nodes from the graph representation.
def clean_same_add(nodes):
    for i in range(len(nodes)):
        nodes[i][0] = i
    idx_map = {}
    new_nodes = []
    for i in range(len(nodes)):
        if nodes[i][1][0] == 255:
            new_nodes.append(nodes[i])
            continue
        if nodes[i][1][0] == 0:
            new_nodes.append(nodes[i])
            idx_map[nodes[i][0]] = len(new_nodes) - 1
            continue
        if nodes[i][1][0] > 0:
            if nodes[i][1][0] == 2:
                if nodes[i][1][1] == nodes[i][1][2] or len(nodes[i][1]) < 4:
                    idx = nodes[i][0] - nodes[i][1][1]
                    idx_map[nodes[i][0]] = idx_map[idx]
                    continue
            new_nodes.append(nodes[i])
            self_idx = len(new_nodes) - 1
            idx_map[nodes[i][0]] = self_idx
            for j in range(nodes[i][1][0]):
                idx = nodes[i][0] - nodes[i][1][1 + j]
                new_nodes[-1][1][1 + j] = self_idx - idx_map[idx]
    return new_nodes


# Convert graph nodes back into a gene string.
def get_gene_by_nodes(nodes):
    params_list = [i[1] for i in nodes]
    units = [",".join([str(param) for param in params]) for params in params_list]
    return "-".join(units)


# Build an adjacency matrix and parameter list from gene units.
def get_adjacency_matrix_with_params(gene: str):
    gene_units = get_units_by_gene(gene)[1:-1]
    matrix = [[0 for _ in range(len(gene_units) + 2)] for _ in range(len(gene_units) + 2)]
    info = ['start']
    pos = 1
    for gene_unit in gene_units:
        gene_unit_params = [int(i) for i in gene_unit.split(',')]
        if gene_unit_params[0] == 1:
            offset = gene_unit_params[1]

            matrix[max(pos - offset, 0)][pos] = 1

            info.append('i:{};t:{};c:{};k:{};d:{};a:{}'.format(gene_unit_params[0],gene_unit_params[2],gene_unit_params[3], gene_unit_params[4], gene_unit_params[5],gene_unit_params[6]))
        elif gene_unit_params[0] == 2:
            offset_a = gene_unit_params[1]
            offset_b = gene_unit_params[2]

            matrix[max(pos - offset_a, 0)][pos] = 1
            matrix[max(pos - offset_b, 0)][pos] = 1
            info.append('i:{};t:{}'.format(gene_unit_params[0],gene_unit_params[3]))
        pos += 1
    matrix[-2][-1] = 1
    info.append('end')
    return matrix, info


# Convert matrix rows into sortable binary signatures.
def get_bin_nums(matrix):
    def get_bin_num(row):
        res = 0
        for i in row:
            res <<= 1
            res += i
        return res

    res = []
    for row in matrix:
        res.append(get_bin_num(row))
    return res


# Remove unused graph nodes while preserving gene parameters.
def clean_matrix_with_params(matrix, params):
    while True:
        temp = []
        nums = get_bin_nums(matrix)
        nums[-1] = 1
        for i in range(len(nums)):
            if nums[i] == 0:
                params[i] = 'delete'
                continue
            temp.append([])
            for j in range(len(nums)):
                if nums[j] == 0:
                    continue
                temp[-1].append(matrix[i][j])
        while 'delete' in params:
            params.remove('delete')
        if len(temp) == len(matrix):
            return temp, params
        else:
            matrix = temp


# Sort the graph matrix and parameters into canonical order.
def sort_matrix_with_params(matrix, params):
    k = 0
    rc_len = len(matrix)
    while k == 0:
        nums = get_bin_nums(matrix)
        for i in range(rc_len):
            if i == rc_len - 1:
                k = 1
                break
            if nums[i] >= nums[i + 1]:
                continue
            else:
                for j in range(rc_len):
                    temp = matrix[i][j]
                    matrix[i][j] = matrix[i + 1][j]
                    matrix[i + 1][j] = temp
                for j in range(rc_len):
                    temp = matrix[j][i]
                    matrix[j][i] = matrix[j][i + 1]
                    matrix[j][i + 1] = temp
                temp = params[i]
                params[i] = params[i + 1]
                params[i + 1] = temp
                break


# Reconstruct a gene string from a graph matrix and parameters.
def get_gene_by_matrix_with_params(matrix, params: List[str]):
    gene_units = ["0"]
    for i in range(1, len(matrix) - 1):
        col = [row[i] for row in matrix]

        param = params[i]
        items = param.split(";")
        input = items[0].split(":")[1]
        input = int(input)

        input_offsets = [i - j for j in range(len(matrix)) if col[j] == 1]

        if input > 1:
            if len(input_offsets) == 1:
                input_offsets = [input_offsets[0], input_offsets[0]]
            itype = items[1].split(":")[1]
            itype = int(itype)

            unit = ",".join([str(j) for j in ([2] + input_offsets + [itype])])
            gene_units.append(unit)
        else:
            t = items[1].split(":")[1]
            c = items[2].split(":")[1]
            k = items[3].split(":")[1]
            d = items[4].split(":")[1]
            a = items[5].split(":")[1]

            unit = ",".join([str(j) for j in ([1] + input_offsets + [t, c, k, d, a])])
            gene_units.append(unit)
    gene_units.append("255")
    return "-".join(gene_units)
