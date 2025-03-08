import importlib
import joblib
import logging
import numpy as np
import os

from maorkov import Sample, Population, Cross

class Simulation:

    def __init__(self, param=None, sim_name="example", print_log=False, log_name="example"):
        # logger
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s][%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.print_log = print_log
        # path
        current = os.path.abspath(__file__)
        current_dir = os.path.dirname(current)
        self.tmp_path = os.path.join(current_dir, "tmp", sim_name)
        self.output_path = os.path.join(current_dir, "output", sim_name)
        for path in [self.tmp_path, self.output_path]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
        self.output_path = os.path.join(self.output_path, log_name + ".log")
        # handle parameters
        self.params = importlib.import_module(f"params.{param}")
        self.total_gene_num = self.params.TOTAL_GENE_NUM
        self.target_gene_num = self.params.TARGET_GENE_NUM
        self.target_gene_idx, self.background_gene_idx = self.get_gene_idx()
        self.max_recombination_rate = self.params.MAX_RECOMBINATION_RATE
        self.init_donor_bg_freq = self.params.INITIAL_DONOR_BACKGROUND_GENE_FREQUENCY
        self.init_receptor_bg_freq = self.params.INITIAL_RECEPTOR_BACKGROUND_GENE_FREQUENCY
        self.recombination_rate = self.get_recombination_rate()
        # init population
        self.initial_donor_num = self.params.INITIAL_DONOR_NUM
        self.initial_receptor_num = self.params.INITIAL_RECEPTOR_NUM
        self.initial_donor_pop, self.initial_receptor_pop = self.init_population()

        self.abstract()

    def log(self, msg=None):
        """
        Logger of maorkov
        :param msg: str
        """
        if self.print_log:
            self.logger.info(msg)

    def write_output(self, msg=None):
        with open(os.path.join(self.output_path), "a") as f:
            f.writelines(msg + "\n")

    def abstract(self):
        """
        Print abstract of simulation
        """
        abstract_str_list = [
            "\nAbstract:",
            f"- Total gene number: {self.total_gene_num}",
            f"- Target gene position: {self.target_gene_idx}",
            f"- Init population target gene frequency:",
            f"  - Donor {self.initial_donor_pop.target_gene_frequency}",
            f"  - Receptor {self.initial_receptor_pop.target_gene_frequency}",
            f"- Init population background gene average frequency:",
            f"  - Donor {self.initial_donor_pop.avg_bg_gene_frequency}",
            f"  - Receptor {self.initial_receptor_pop.avg_bg_gene_frequency}",
        ]
        abstract_str = '\n'.join(abstract_str_list)
        self.log(abstract_str)

    def get_gene_idx(self):
        """
        Get the index of target gene and background gene.
        :return: np.array, np.array
        """
        target_gene_idx_file = os.path.join(self.tmp_path, "target_gene_idx.pkl")
        background_gene_idx_file = os.path.join(self.tmp_path, "background_gene_idx.pkl")
        if os.path.exists(target_gene_idx_file) and os.path.exists(background_gene_idx_file):
            self.log(f'Reuse existing target gene idx...')
            return joblib.load(target_gene_idx_file), joblib.load(background_gene_idx_file)
        else:
            gene_idx = np.arange(self.total_gene_num)
            np.random.shuffle(gene_idx)
            target_gene_idx = gene_idx[:self.target_gene_num]
            background_gene_idx = gene_idx[self.target_gene_num:]
            joblib.dump(target_gene_idx, target_gene_idx_file)
            joblib.dump(background_gene_idx, background_gene_idx_file)
            return target_gene_idx, background_gene_idx

    def get_recombination_rate(self):
        """
        Get recombination rate
        The recombination rate is randomly generated from 0 to the max_recombination_rate
        :return: np.array
        """
        recombination_rate_file = os.path.join(self.tmp_path, "recombination_rate.pkl")
        if os.path.exists(recombination_rate_file):
            self.log(f'Reuse existing recombination rate...')
            return joblib.load(recombination_rate_file)
        else:
            # save
            recombination_rate = np.random.rand(self.total_gene_num, 1) * self.max_recombination_rate
            joblib.dump(recombination_rate, recombination_rate_file)
            return recombination_rate

    def init_population(self):
        """
        Initialize population
        """
        initial_donor_pop_file = os.path.join(self.tmp_path, "initial_donor_pop.pkl")
        initial_receptor_pop_file = os.path.join(self.tmp_path, "initial_receptor_pop.pkl")
        # reuse if exist
        if os.path.exists(initial_donor_pop_file) and os.path.exists(initial_receptor_pop_file):
            self.log(f'Reuse existing initial population...')
            return joblib.load(initial_donor_pop_file), joblib.load(initial_receptor_pop_file)
        # init population
        initial_donor_pop = Population(
            pop_name='initial_donor_pop',
            gene_num=self.total_gene_num,
            target_gene_idx=self.target_gene_idx,
            background_gene_idx=self.background_gene_idx
        )
        initial_receptor_pop = Population(
            pop_name='initial_receptor_pop',
            gene_num=self.total_gene_num,
            target_gene_idx=self.target_gene_idx,
            background_gene_idx=self.background_gene_idx
        )
        initial_donor_pop.create(
            number=self.initial_donor_num,
            method="random",
            distribution=self.init_donor_bg_freq
        )
        initial_receptor_pop.create(
            number=self.initial_receptor_num,
            method="random",
            distribution=self.init_receptor_bg_freq
        )
        # edit donor population, target gene all be 2
        initial_donor_pop.edit_all(position=self.target_gene_idx, genotype=2)
        # edit receptor population, target gene all be 0
        initial_receptor_pop.edit_all(position=self.target_gene_idx, genotype=0)
        # save
        joblib.dump(initial_donor_pop, initial_donor_pop_file)
        joblib.dump(initial_receptor_pop, initial_receptor_pop_file)
        return initial_donor_pop, initial_receptor_pop


if __name__ == "__main__":
    sim = Simulation(param="example")
