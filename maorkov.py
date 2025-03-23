import numpy as np
import random
import time
import uuid

from collections import Counter
from tqdm import tqdm


class Sample:
    """
    sample class
    """
    def __init__(
            self,
            sample_id=None,
            sex='M',
            father=None,
            mother=None,
            family=None,
            pedigree=None,
            gene=None,
            gene_num=None
    ):
        """
        :param sample_id:
        :param sex:
        :param father:
        :param mother:
        :param family:
        :param pedigree:
        :param gene:
        :param gene_num:
        """
        self.id = sample_id or str(uuid.uuid4())
        self.sex = sex
        self.mother = mother
        self.father = father
        self.family = family or father or self.id
        self.pedigree = pedigree
        self.gene = gene
        self.gene_num = gene_num

    def __str__(self):
        sample_info = [
            f"id: {self.id}",
            f"sex: {self.sex}",
            f"mother: {self.mother}",
            f"father: {self.father}",
            f"family: {self.family}",
        ]
        return ",".join(sample_info)

    def is_male(self):
        """
        Check if the sample is male
        :return: True or False
        """
        return self.sex == 'M'

    def get_genotype(self, position=None):
        """
        Get sample's genotype at position
        :param position: int
        :return: array of [int, int]
        """
        return self.gene[position]

    def get_genotype_sum(self, position=None):
        """
        Get sample's genotype sum at position
        :param position: int
        :return: sum of [0(Recessive homozygote) 1(Heterozygote) or 2(Dominant homozygote)]
        """
        return np.sum(self.get_genotype(position=position))

    def edit_genotype(self, position=None, genotype=2):
        """
        Edit genotype at position
        :param position: list of int
        :param genotype: 0(Recessive homozygote) 1(Heterozygote) or 2(Dominant homozygote)
        """
        for idx in position:
            if genotype == 0:
                for p in range(2):
                    self.gene[idx, p] = 0
            if genotype == 2:
                for p in range(2):
                    self.gene[idx, p] = 1

    def initial_gene_with_zeros(self):
        """
        Initial genotype of sample with all 0
        """
        self.gene = np.zeros((self.gene_num, 2), dtype=int)

    def initial_gene_with_distribution(self, distribution=None):
        """
        Initial genotype of sample with distribution
        """
        values = [1, 0]
        self.gene = np.random.choice(values, size=(self.gene_num, 2), p=distribution)

    def get_sample_position_gene_num(self, position=None):
        """
        Get genotype number of sample at position
        :param position: list of int
        :return: numbers of 0(Recessive homozygote) 1(Heterozygote) or 2(Dominant homozygote)
        """
        gene_genotype = self.get_genotype(position=position)
        domi_homo, hetero, rece_homo = 0, 0, 0
        for genotype in gene_genotype:
            if sum(genotype) == 2:
                domi_homo += 1
            if sum(genotype) == 1:
                hetero += 1
            if sum(genotype) == 0:
                rece_homo += 1
        assert (domi_homo + hetero + rece_homo) == len(position)
        return domi_homo, hetero, rece_homo


class Population:
    """
    Population class
    """
    def __init__(
            self,
            sample_list=None,
            pop_name=None,
            gene_num=None,
            target_gene_idx=None,
            background_gene_idx=None
    ):
        self.sample_list = sample_list or []
        self.family_count = None
        self.pop_name = pop_name
        self.gene_num = gene_num
        self.target_gene_idx = target_gene_idx
        self.background_gene_idx = background_gene_idx

    def add(self, sample=None):
        """
        Add sample to population
        :param sample: instance of Sample
        """
        assert isinstance(sample, Sample)
        self.sample_list.append(sample)

    def create(self, number=None, method="zero", distribution=None):
        """
        Create population with given number and method
        :param number: int (must be an even number)
        :param method: zero or random
        :param distribution: required if method is random
        """
        if len(self.sample_list) > 0:
            raise Exception('Population already created, use append() to add samples')
        else:
            print(f'Creating population [{self.pop_name}] with {number} samples...')
            for i in range(number):
                sample_sex = 'M' if i % 2 == 0 else 'F'
                sample = Sample(
                    sex=sample_sex,
                    gene_num=self.gene_num
                )
                if method == "zero":
                    sample.initial_gene_with_zeros()
                if method == "random":
                    sample.initial_gene_with_distribution(distribution=distribution)
                self.add(sample)

    def get_father(self):
        return [sample for sample in self.sample_list if sample.sex == "M"]

    def get_mother(self):
        return [sample for sample in self.sample_list if sample.sex == "F"]

    def shuffle(self):
        """
        Shuffle samples in population
        """
        random.shuffle(self.sample_list)

    def half_divide_pop(self):
        self.shuffle()
        half_pop_size = int(len(self.sample_list) / 2)
        pop1 = Population(
            sample_list=self.sample_list[:half_pop_size],
            pop_name="tmp",
            gene_num=self.gene_num,
            target_gene_idx=self.target_gene_idx,
            background_gene_idx=self.background_gene_idx
        )
        pop2 = Population(
            sample_list=self.sample_list[half_pop_size:],
            pop_name="tmp",
            gene_num=self.gene_num,
            target_gene_idx=self.target_gene_idx,
            background_gene_idx=self.background_gene_idx
        )
        return pop1, pop2

    def __getitem__(self, idx=None):
        """
        Get sample by idx
        :param idx: index
        :return: the index sample
        """
        return self.sample_list[idx]

    def __len__(self):
        """
        Get population size
        :return: int
        """
        return len(self.sample_list)

    def edit_all(self, position=None, genotype=2):
        """
        Edit all samples in population
        :param position: position idx list
        :param genotype: 0 1 or 2
        """
        for sample in self.sample_list:
            sample.edit_genotype(position=position, genotype=genotype)

    @property
    def all_gene_frequency(self):
        """
        Get frequency of all gene in population
        :return: np.array
        """
        gene_frequency_list = []
        for gene in range(self.gene_num):
            position = np.array([gene])
            gene_frequency = 0
            for sample in self.sample_list:
                gene_frequency += sample.get_genotype_sum(position=position)
            gene_frequency = gene_frequency / (2 * len(self.sample_list))
            gene_frequency_list.append(gene_frequency)
        return np.array(gene_frequency_list)

    @property
    def background_gene_frequency(self):
        """
        Get frequency of background gene in population
        :return: np.array
        """
        return self.all_gene_frequency[self.background_gene_idx]

    @property
    def avg_bg_gene_frequency(self):
        """
        Get frequency of background gene in population
        :return: np.array
        """
        return np.average(self.background_gene_frequency)

    @property
    def bg_gene_all_positive_percent(self):
        """
        Get percentage of background gene positive(1 or 2) in population
        :return:
        """
        bg_gene_all_positive_list = []
        for sample in self.sample_list:
            bg_gene_all_positive = 1
            for gene in range(self.gene_num):
                position = np.array([gene])
                genotype_sum = sample.get_genotype_sum(position=position)
                if genotype_sum == 0:
                    bg_gene_all_positive = 0
            bg_gene_all_positive_list.append(bg_gene_all_positive)
        return sum(bg_gene_all_positive_list) / len(bg_gene_all_positive_list)

    @property
    def target_gene_frequency(self):
        """
        Get frequency of target gene in population
        :return: np.array
        """
        return self.all_gene_frequency[self.target_gene_idx]

    @property
    def avg_target_gene_frequency(self):
        """
        Get frequency of background gene in population
        :return: np.array
        """
        return np.average(self.target_gene_frequency)

    def sort_by_target(self, reverse=True):
        """
        Sort population by target gene sum
        """
        self.sample_list = sorted(
            self.sample_list,
            reverse=reverse,
            key=lambda sample: sample.get_genotype_sum(position=self.target_gene_idx)
        )

    def sort_by_background(self, reverse=True):
        """
        Sort population by background gene sum
        """
        self.sample_list = sorted(
            self.sample_list,
            reverse=reverse,
            key=lambda sample: sample.get_genotype_sum(position=self.background_gene_idx)
        )

    def __str__(self):
        """
        Population abstract
        """
        abstract_list = [
            f"\n{self.pop_name}:",
            f"Population size: {len(self)}",
            f"Target gene frequency: {self.target_gene_frequency}",
            f"Background gene average frequency: {self.avg_bg_gene_frequency}"
        ]
        return '\n- '.join(abstract_list)


class Cross(object):
    """
    Cross class
    """
    def __init__(
            self,
            recombination_rate=None,
            cross_sex_ratio=10,
            pregnant_per_litter=10,
            breed_type='AvoidSameLitter',
            sort_priority="TargetGeneFirst",
            gene_num=None,
            target_gene_idx=None,
            background_gene_idx=None
    ):
        """
        :param recombination_rate:
        :param cross_sex_ratio: number of female with one male
        :param pregnant_per_litter: number of progeny per litter
        :param breed_type: AvoidSameLitter, Normal
        :param sort_priority: TargetGeneFirst or BackgroundGeneFirst, prioritized sorting method  before crossing
        """
        self.recombination_rate = recombination_rate
        self.breed_type = breed_type
        self.cross_sex_ratio = cross_sex_ratio
        self.pregnant_per_litter = pregnant_per_litter
        self.sort_priority = sort_priority
        self.gene_num = gene_num
        self.target_gene_idx = target_gene_idx
        self.background_gene_idx = background_gene_idx

    @staticmethod
    def _gamete(l_array, j_array):
        """
        Get gamete
        :param l_array: (g, 2)
        :param j_array: (g, 1)
        :return: (g, 1)
        """
        assert l_array.shape[0] == j_array.shape[0]
        n = l_array.shape[0]
        g = np.zeros((n, 1))
        for i in range(n):
            g[i, 0] = l_array[i][j_array[i]]
        return g

    def _breed(self, father=None, mother=None):
        """
        Breed between father and mother
        :param father: instance of Sample
        :param mother: instance of Sample
        :return: list of samples
        """
        # breed number must be an even number, otherwise the male and female cannot be divided equally.
        assert self.pregnant_per_litter % 2 == 0
        assert isinstance(father, Sample)
        assert isinstance(mother, Sample)
        l1 = father.gene
        l2 = mother.gene
        recombine = self.recombination_rate[1:]  # The first gene does not require recombination
        assert l1.shape[0] == l2.shape[0] == recombine.shape[0] + 1
        assert l1.shape[1] == l2.shape[1] == 2

        progeny_list = []
        n = l1.shape[0]
        # get a list of random number to simulate recombination
        prob = np.random.rand(n, 2, self.pregnant_per_litter)
        # get progeny
        for ki in range(self.pregnant_per_litter):
            # get J matrix of l1
            j1 = np.zeros((n, 1), dtype=int)
            for i in range(n):
                prob_ik1 = prob[i, 0, ki]
                if i == 0:
                    j1[i] = 1 if prob_ik1 >= 0.5 else 0
                else:
                    j1[i] = j1[i - 1] if prob_ik1 >= recombine[i - 1] else (1 - j1[i - 1])
            # get gamete of l1
            g1 = self._gamete(l1, j1)

            # get J matrix of l2
            j2 = np.zeros((n, 1), dtype=int)
            for i in range(n):
                prob_ik2 = prob[i, 1, ki]
                if i == 0:
                    j2[i] = 1 if prob_ik2 >= 0.5 else 0
                else:
                    j2[i] = j2[i - 1] if prob_ik2 >= recombine[i - 1] else (1 - j2[i - 1])
            # get gamete of l2
            g2 = self._gamete(l2, j2)

            # create a pregnant by combining two gametes
            p = np.hstack((g1, g2))
            p_sex = 'M' if ki % 2 == 0 else 'F'
            progeny = Sample(
                sex=p_sex,
                father=father.id,
                mother=mother.id,
                family=father.family,
                gene=p
            )
            progeny_list.append(progeny)

        return progeny_list

    def breed(self, father=None, mother=None):
        """
        Breed between a father and a mother
        :param father: instance of Sample
        :param mother: instance of Sample
        :return: list
        """
        assert father.sex == 'M'
        assert mother.sex == 'F'
        # avoid same litter sample crossing
        if self.breed_type == 'AvoidSameLitter':
            if father.father == mother.father and father.mother == mother.mother:
                return None
            else:
                pass
        return self._breed(father=father, mother=mother)

    def cross(self, father_pop=None, mother_pop=None, father_num=None, pop_name=None):
        """
        Cross between father population and mother population
        :param father_pop: father population
        :param mother_pop: mother population
        :param father_num: father number for crossing
        :param pop_name: name of progeny population
        :return: progeny pop
        """
        assert isinstance(father_pop, Population)
        assert isinstance(mother_pop, Population)
        # sort first
        if self.sort_priority == "TargetGeneFirst":
            father_pop.sort_by_background()
            mother_pop.sort_by_background()
            father_pop.sort_by_target()
            mother_pop.sort_by_target()
        else:
            father_pop.sort_by_target()
            mother_pop.sort_by_target()
            father_pop.sort_by_background()
            mother_pop.sort_by_background()
        # get father and mother from pops
        father_list = father_pop.get_father()
        mother_list = mother_pop.get_mother()
        # check mother number if is enough
        father_num = father_num or len(father_list)
        if len(mother_list) < father_num * self.cross_sex_ratio:
            raise Exception(f"No enough mother: mother number {len(mother_list)}, father number {father_num}")
        progeny_list = []
        mother_used_set = set()
        for father in father_list[:father_num]:
            mother_used_count = 0
            for mother in mother_list:
                # cross_sex_ratio mother per father
                if mother_used_count >= self.cross_sex_ratio:
                    break
                # avoid mother reuse
                if mother.id not in mother_used_set:
                    progeny = self.breed(father=father, mother=mother)
                    if progeny is not None:
                        mother_used_set.add(mother.id)
                        mother_used_count += 1
                        progeny_list += progeny
        expected_progeny_number = father_num * self.cross_sex_ratio * self.pregnant_per_litter
        if len(progeny_list) != expected_progeny_number:
            raise Exception(f"Cross number incorrect: progeny number {len(progeny_list)}, "
                            f"expected number {expected_progeny_number}")

        new_pop = Population(
            sample_list=progeny_list,
            pop_name=pop_name,
            gene_num=self.gene_num,
            target_gene_idx=self.target_gene_idx,
            background_gene_idx=self.background_gene_idx
        )
        return new_pop

    def self_cross(self, pop=None, father_num=None, pop_name=None):
        """
        Cross in population itself
        :param pop: population
        :param father_num: father number for crossing
        :param pop_name: population name
        :return: population
        """
        return self.cross(
            father_pop=pop,
            mother_pop=pop,
            father_num=father_num,
            pop_name=pop_name
        )

