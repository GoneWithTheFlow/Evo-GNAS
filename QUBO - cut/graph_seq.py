# Build an adjacency matrix from gene units.
def get_adjacency_matrix(gene_units):
    matrix = [[0 for _ in range(len(gene_units) + 2)] for _ in range(len(gene_units) + 2)]
    pos = 1
    for gene_unit in gene_units:
        gene_unit_params = [int(i) for i in gene_unit.split(',')]
        if gene_unit_params[0] == 1:
            offset = gene_unit_params[1]
            matrix[max(pos - offset, 0)][pos] = 1
        elif gene_unit_params[0] == 2:
            offset_a = gene_unit_params[1]
            offset_b = gene_unit_params[2]
            matrix[max(pos - offset_a, 0)][pos] = 1
            matrix[max(pos - offset_b, 0)][pos] = 1
        pos += 1
    matrix[-2][-1] = 1
    return matrix


# Build an adjacency matrix and metadata list from gene units.
def get_adjacency_matrix_with_info(gene_units):
    matrix = [[0 for _ in range(len(gene_units) + 2)] for _ in range(len(gene_units) + 2)]

    info = ['start']
    pos = 1
    for gene_unit in gene_units:
        gene_unit_params = [int(i) for i in gene_unit.split(',')]
        if gene_unit_params[0] == 1:
            offset = gene_unit_params[1]

            matrix[max(pos - offset, 0)][pos] = 1

            activation_dict = {0: 'None', 1: 'ReLU', 2: 'eLU', 3: 'Leaky_ReLu'}

            info.append(f'GNN_t{gene_unit_params[2]}c{gene_unit_params[3]}k{gene_unit_params[5]}g1_{activation_dict[gene_unit_params[6]]}')
        elif gene_unit_params[0] == 2:
            offset_a = gene_unit_params[1]
            offset_b = gene_unit_params[2]

            matrix[max(pos - offset_a, 0)][pos] = 1
            matrix[max(pos - offset_b, 0)][pos] = 1

            type_dict = {2: 'add', 3: 'concat'}

            info.append('{}'.format(type_dict[gene_unit_params[3]]))
        pos += 1
    matrix[-2][-1] = 1
    info.append('end')
    return matrix, info


# Remove unused nodes from an adjacency matrix.
def clean_matrix(matrix):
    while True:
        temp = []
        nums = get_bin_nums(matrix)
        nums[-1] = 1
        for i in range(len(nums)):
            if nums[i] == 0:
                continue
            temp.append([])
            for j in range(len(nums)):
                if nums[j] == 0:
                    continue
                temp[-1].append(matrix[i][j])
        if len(temp) == len(matrix):
            return temp
        else:
            matrix = temp


# Remove unused nodes while preserving node metadata.
def clean_matrix_with_info(matrix, info):
    while True:
        temp = []
        nums = get_bin_nums(matrix)
        nums[-1] = 1
        for i in range(len(nums)):
            if nums[i] == 0:
                info[i] = 'delete'
                continue
            temp.append([])
            for j in range(len(nums)):
                if nums[j] == 0:
                    continue
                temp[-1].append(matrix[i][j])
        while 'delete' in info:
            info.remove('delete')
        if len(temp) == len(matrix):
            return temp, info
        else:
            matrix = temp


# Sort an adjacency matrix into canonical order.
def sort_matrix(matrix):
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
                break


# Sort an adjacency matrix and metadata into canonical order.
def sort_matrix_with_info(matrix, info):
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
                temp = info[i]
                info[i] = info[i + 1]
                info[i + 1] = temp
                break


# Convert one adjacency row into a binary signature.
def get_bin_num(row):
    res = 0
    for i in row:
        res <<= 1
        res += i
    return res


# Convert matrix rows into sortable binary signatures.
def get_bin_nums(matrix):
    res = []
    for row in matrix:
        res.append(get_bin_num(row))
    return res


# Serialize an adjacency matrix into a sequence string.
def matrix_seq(matrix):
    res = []
    for i in get_bin_nums(matrix):
        if i > 0:
            res.append(str(i))
    return '-'.join(res)


# Convert a gene graph into a sequence string.
def gene_graph_seq(gene_units):
    adj_matrix = get_adjacency_matrix(gene_units)
    adj_matrix = clean_matrix(adj_matrix)
    sort_matrix(adj_matrix)
    adj_matrix = clean_matrix(adj_matrix)
    return matrix_seq(adj_matrix)


# Convert a gene graph into a sequence string with metadata.
def gene_graph_seq_with_info(gene_units):
    adj_matrix, info = get_adjacency_matrix_with_info(gene_units)
    adj_matrix, info = clean_matrix_with_info(adj_matrix, info)
    sort_matrix_with_info(adj_matrix, info)
    adj_matrix, info = clean_matrix_with_info(adj_matrix, info)
    return matrix_seq(adj_matrix), info


# Convert a sequence string back into an adjacency matrix.
def seq_to_matrix(seq: str):
    row_idxs = [int(i) for i in seq.split('-')]
    matrix = []
    for row_idx in row_idxs:
        k = row_idx
        row = []
        while k > 0:
            row.append(k & 1)
            k >>= 1
        row.append(0)
        if len(matrix) > 0:
            for _ in range(len(matrix[0]) - len(row)):
                row.append(0)
        row.reverse()
        matrix.append(row)
    matrix.append([0 for _ in range(len(matrix[0]))])
    return matrix


# Prepare a gene before graph-sequence conversion.
def pre_gene_for_seq(gene):
    return gene.split("-")[1:-1]


# Format sequence metadata as a string.
def get_seq_info_str(seq, info):
    return seq + '|' + '-'.join(info)
