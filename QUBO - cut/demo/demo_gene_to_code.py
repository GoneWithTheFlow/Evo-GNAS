import architecture_codegen
from evolutionary_search import fix_gene_params


# Convert gene unit parameters into generated model code.
def gene_to_code(gene):
    gene_units = gene.split('-')

    gene_units_params = [unit.split(',') for unit in gene_units]

    gene_units_params_fixed = fix_gene_params(gene_units_params)
    print(gene_units_params_fixed)
    code, gene_code_length = architecture_codegen.gene_to_code(gene_units_params_fixed)
    return code, gene_code_length

gene = '0-1,1,2,16,4,30,2-1,1,3,64,4,0,1-1,1,1,16,4,0,1-2,1,3,2-1,1,6,16,3,0,1-2,1,5,2-1,2,5,16,4,0,1-2,1,2,3-1,1,1,1,4,0,0-255'
code, gene_code_length = gene_to_code(gene)
print(code)
