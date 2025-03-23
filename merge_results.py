import joblib
import numpy as np
from tqdm import tqdm

from params.example import FATHER_BUDGET, RECURSIVE_FATHER_NUM_LIST

output_dir = "output/matrix"
result_dir = "output/data/five_gene"
result_name = "five_gene"
repeat_num = 256

stage_num = 10
start = 0
end = 1.0
progresses = []
interval = (end - start) / stage_num
for i in range(stage_num):
    progresses.append((start + i * interval, start + (i + 1) * interval))
progresses.append((end, float('inf')))
print('Stages:', progresses)
def get_progress_index(s):
    for j in range(len(progresses)):
        start_j, end_j = progresses[j]
        if start_j <= s < end_j:
            return j

transition_matrix_dict = {a * 100: np.zeros((stage_num + 1, stage_num + 1)) for a in RECURSIVE_FATHER_NUM_LIST}
for r_id in tqdm(range(repeat_num)):

    result_file = f"{result_dir}/{result_name}_{r_id}.log"
    with open(result_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            status, _, target_gene_list, background_gene_list, father_list = line.strip().split(";")
            target_gene_list = target_gene_list.split(",")
            background_gene_list = background_gene_list.split(",")
            father_list = father_list.split(",")
            assert len(target_gene_list) == len(background_gene_list) == (len(father_list) + 1)
            for i in range(len(father_list)):
                start_gene = float(target_gene_list[i])
                end_gene = float(target_gene_list[i + 1])
                father_num = int(father_list[i])
                start_stage = get_progress_index(start_gene)
                end_stage = get_progress_index(end_gene)
                transition_matrix_dict[father_num * 100][start_stage, end_stage] += 1

for a in transition_matrix_dict:
    print(transition_matrix_dict[a].shape)
    transition_matrix_dict[a] = transition_matrix_dict[a] / np.sum(transition_matrix_dict[a])
    transition_matrix_dict[a] = transition_matrix_dict[a] / transition_matrix_dict[a].sum(axis=1).reshape(-1, 1)
    transition_matrix_dict[a] = np.nan_to_num(transition_matrix_dict[a])
    np.savetxt(f'{output_dir}/W_{a}.csv', transition_matrix_dict[a], fmt='%f', delimiter=',')

joblib.dump(transition_matrix_dict, 'output/transition_matrix.pkl')
