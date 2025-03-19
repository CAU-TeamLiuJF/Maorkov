import os
import time
import multiprocessing

import joblib

from maorkov import Cross
from simulation import Simulation


class Example(Simulation):

    def __init__(self, param=None, sim_name="example", print_log=False, log_name="example"):
        super().__init__(param=param, sim_name=sim_name, print_log=print_log, log_name=log_name)
        self.cross_sex_ratio = self.params.CROSS_SEX_RATION
        self.pregnant_per_litter = self.params.PREGNANT_PER_LITTER
        self.inbred_father_num = self.params.INBRED_FATHER_NUM
        self.back_cross_father_num = self.params.BACK_CROSS_FATHER_NUM
        self.recursive_father_num_list = self.params.RECURSIVE_FATHER_NUM_LIST
        self.recursive_max_generation = self.params.RECURSIVE_MAX_GENERATION
        self.father_budget = self.params.FATHER_BUDGET
        self.f1_pop, self.f1_receptor_inbred = self.get_f1()
        self.log(self.f1_pop)
        self.log(self.f1_receptor_inbred)
        self.recursive_cross = Cross(
            breed_type="AvoidSameLitter",  # recursive cross use AvoidSameLitter strategy
            recombination_rate=self.recombination_rate,
            cross_sex_ratio=self.cross_sex_ratio,
            pregnant_per_litter=self.pregnant_per_litter,
            gene_num=self.total_gene_num,
            target_gene_idx=self.target_gene_idx,
            background_gene_idx=self.background_gene_idx,
        )

    def get_f1(self):
        f1_pop_file = os.path.join(self.tmp_path, "f1_pop.pkl")
        f1_receptor_inbred_file = os.path.join(self.tmp_path, "f1_receptor_inbred.pkl")
        # reuse if exist
        if os.path.exists(f1_pop_file) and os.path.exists(f1_receptor_inbred_file):
            self.log(f"Reuse existing f1 pop...")
            return joblib.load(f1_pop_file), joblib.load(f1_receptor_inbred_file)
        # f1 cross
        f1_cross = Cross(
            breed_type=None, # f1 do not use AvoidSameLitter strategy
            recombination_rate=self.recombination_rate,
            cross_sex_ratio=self.cross_sex_ratio,
            pregnant_per_litter=self.pregnant_per_litter,
            gene_num=self.total_gene_num,
            target_gene_idx=self.target_gene_idx,
            background_gene_idx=self.background_gene_idx,
        )
        init_receptor_pop1, init_receptor_pop2 = self.initial_receptor_pop.half_divide_pop()
        f1_pop = f1_cross.cross(
            father_pop=self.initial_donor_pop,
            mother_pop=init_receptor_pop1,
            pop_name="f1_pop"
        )
        receptor_inbred_pop = f1_cross.self_cross(
            pop=init_receptor_pop2,
            father_num=self.inbred_father_num,
            pop_name="receptor_inbred_pop"
        )
        joblib.dump(f1_pop, f1_pop_file)
        return f1_pop, receptor_inbred_pop

    def back_cross(self):
        back_cross_pop_file = os.path.join(self.tmp_path, "back_cross_pop.pkl")
        back_cross_inbred_pop_file = os.path.join(self.tmp_path, "back_cross_inbred_pop.pkl")
        # reuse if exist
        if os.path.exists(back_cross_pop_file) and os.path.exists(back_cross_inbred_pop_file):
            return joblib.load(back_cross_pop_file), joblib.load(back_cross_inbred_pop_file)
        back_cross = Cross(
            breed_type="AvoidSameLitter",  # back cross use AvoidSameLitter strategy
            recombination_rate=self.recombination_rate,
            cross_sex_ratio=self.cross_sex_ratio,
            pregnant_per_litter=self.pregnant_per_litter,
            gene_num=self.total_gene_num,
            target_gene_idx=self.target_gene_idx,
            background_gene_idx=self.background_gene_idx,
        )
        # back cross twice
        for i in range(1):
            back_cross_receptor_pop1, back_cross_receptor_pop2 = self.f1_receptor_inbred.half_divide_pop()
            self.f1_pop = back_cross.cross(
                father_pop=self.f1_pop,
                mother_pop=back_cross_receptor_pop1,
                father_num=self.back_cross_father_num,
                pop_name=f"back_cross_pop_{i + 1}"
            )
            self.f1_receptor_inbred = back_cross.self_cross(
                pop=back_cross_receptor_pop2,
                father_num=self.inbred_father_num,
                pop_name=f"back_cross_inbred_pop_{i + 1}"
            )
            self.log(self.f1_pop)
            self.log(self.f1_receptor_inbred)
            joblib.dump(self.f1_pop, back_cross_pop_file)
            joblib.dump(self.f1_receptor_inbred, back_cross_inbred_pop_file)
        return self.f1_pop, self.f1_receptor_inbred

    def check_success(
            self,
            generation=1,
            target_gene_freq=None,
            father_num_list=None
    ):
        if generation > self.recursive_max_generation:
            return 'Failure(GenerationExceeded)'
        if sum(father_num_list) > self.father_budget:
            return 'Failure(FatherBudgetExceeded)'
        if target_gene_freq >= 1:
            return 'Success'
        return 'Continue'

    def run(self):
        back_cross_pop, back_cross_inbred_pop = self.back_cross()
        self.recursion(pop=back_cross_pop)

    def recursion(
            self,
            pop=None,
            generation=1,
            target_freq_list=None,
            bg_freq_list=None,
            father_num_list=None
    ):
        if target_freq_list is None:
            target_freq_list = [pop.avg_target_gene_frequency]
        if bg_freq_list is None:
            bg_freq_list = [pop.avg_bg_gene_frequency]
        if father_num_list is None:
            father_num_list = []
        for father_num in self.recursive_father_num_list:
            progeny_pop = self.recursive_cross.self_cross(
                pop=pop,
                father_num=father_num,
                pop_name=f"recursive_cross_pop_{generation}"
            )
            next_generation = generation + 1
            target_gene_freq = progeny_pop.avg_target_gene_frequency
            background_gene_freq = progeny_pop.avg_bg_gene_frequency
            next_target_freq_list = target_freq_list + [target_gene_freq]
            next_bg_freq_list = bg_freq_list + [background_gene_freq]
            next_father_num_list = father_num_list + [father_num]
            check_success = self.check_success(
                generation=next_generation,
                target_gene_freq=target_gene_freq,
                father_num_list=next_father_num_list
            )
            log_msg = [
                f"Generation {next_generation}: ",
            ]
            log_msg[0] += f"{check_success}."
            log_msg.append(f"Target freq list: {next_target_freq_list}.")
            log_msg.append(f"Background freq list: {next_bg_freq_list}.")
            log_msg.append(f"Father number list: {next_father_num_list}.")
            log_msg.append("=" * 50)
            self.log("\n".join(log_msg))
            if check_success == 'Success' or 'Failure' in check_success:
                self.write_output(";".join([
                    check_success,
                    str(next_generation),
                    ",".join(map(str, next_target_freq_list)),
                    ",".join(map(str, next_bg_freq_list)),
                    ",".join(map(str, next_father_num_list))
                ]))
            elif check_success == 'Continue':
                self.recursion(
                    pop=progeny_pop,
                    generation=next_generation,
                    target_freq_list=next_target_freq_list,
                    bg_freq_list=next_bg_freq_list,
                    father_num_list=next_father_num_list
                )
            else:
                raise RuntimeError(check_success)


def single_thread(t_id=0):
    # print_log = t_id == 0
    five_gene = Example(
        param="example",
        sim_name="five_gene",
        print_log=True,
        log_name=f"five_gene_{t_id}"
    )
    five_gene.run()


if __name__ == "__main__":
    # create common files by one thread first
    single_thread()
    # then run multi thread simulation
    processes = []
    for i in range(1, 256):
        single_thread(t_id=i)
        time.sleep(10)
        print(f"Process {i} started")
        p = multiprocessing.Process(target=single_thread, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()  # 等待所有进程结束
    print("所有进程运行完成")