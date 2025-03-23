import joblib
import numpy as np


class BackwardInduction(object):

    # 每头种猪的收益
    revenue_per_breeding_pig = 10000
    # 每头肉猪的收益
    revenue_per_pig = 2600
    # 每头猪的养殖成本
    cost_per_pig = 2000
    # 每代所需的固定成本
    stand_by_cost = 200000
    # 贴现因子
    time_lambda = 1

    # 状态数量
    stages_num = 11
    # 产仔数量
    a_list = [300, 400, 500, 600, 700]
    # 预设最大代数
    max_iteration = 6
    # 预设目标完成奖励
    success_reward = 2000000

    def __init__(self):
        self.transition_matrix = joblib.load('output/transition_matrix.pkl')
        for a in self.a_list:
            self.transition_matrix[a] = self.transition_matrix[a]
        self.u_matrix = {t: {
            stage: (None, None) for stage in range(self.stages_num)
        } for t in range(1, self.max_iteration + 1)}

    # f(a, st, st')
    def f(self, a, s_start, s_end):
        return s_end * 100 * self.revenue_per_breeding_pig + self.revenue_per_pig * a

    def c(self, a):
        return self.cost_per_pig * a

    def K(self, t=None):
        return self.stand_by_cost + 100000

    def F(self, s_start, a):
        total = 0
        for s_end in range(self.stages_num):
            p = self.transition_matrix[a][s_start, s_end]
            total += p * self.f(a, s_start, s_end)
        return total

    def r(self, s_start, a, t):
        return self.F(s_start, a) - self.c(a) - self.K(t)

    def u_star(self, t, s_start):
        if s_start == (self.stages_num - 1):
            return self.success_reward, None
        elif t == self.max_iteration:
            return 0, None
        else:
            u_max = 0
            a_max = 0
            for a in self.a_list:
                u_total = 0
                for s_end in range(self.stages_num):
                    p = self.transition_matrix[a][s_start, s_end]
                    result_pre = self.u_matrix[t + 1][s_end]
                    if result_pre[0] is None:
                        u_star_next, _ = self.u_star(t + 1, s_end)
                        u_total += p * u_star_next
                        self.u_matrix[t + 1][s_end] = (u_star_next, None)
                    else:
                        u_star_next, _ = result_pre
                        u_total += p * u_star_next
                        self.u_matrix[t + 1][s_end] = (u_star_next, None)
                u_a = self.r(s_start, a, t) + self.time_lambda * u_total
                if a == self.a_list[0]:
                    u_max = u_a
                    a_max = a
                if u_a > u_max:
                    u_max = u_a
                    a_max = a
            # print(t, u_max, a_max)
            return u_max, a_max

    def u_star_2(self, t, s_start):
        pass


if __name__ == '__main__':
    strategy = {i: {} for i in range(11)}
    mao = BackwardInduction()
    for t_start in range(6, 0, -1):
        output = f'{t_start}'
        # print(t_start)
        for s in range(0, 11):
            st = mao.u_star(t_start, s)
            # print(s, st)
            output += f',{st[-1]}'
            strategy[t_start][s] = st[-1]
        print(output)
    joblib.dump(strategy, 'strategy.pkl')
