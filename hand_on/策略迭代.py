import copy


class CliffWalkingEnv:
    """ 悬崖漫步环境"""

    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P


class PolicyIteration:
    def __init__(self, env=CliffWalkingEnv(), theta=0.001, gamma=0.9, alpha=0.5):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.v = [0] * env.ncol * env.nrow
        self.q = [[0] * 4 for _ in range(env.ncol * env.nrow)]
        self.pi = [[0.25] * 4 for _ in range(env.ncol * env.nrow)]
        self.cnt = 0
        self.alpha = alpha

    # 策略评估是完成了对于状态价值函数V的更新
    # new_v = v + sum(p(s,a) * (r(s,a) + sum(gamma * v(s,a'))))
    # 第一个sum是：对于状态s下，可能采取的动作a的sum
    # 第二个sum是：对于状态s下采取了动作a，下一时刻可能的状态s‘ （其实对于悬崖迷宫而言，下一时刻是可以确定的）

    # 对于策略评估，虽然是对价值函数的评估而不是策略的更新，本质上也是迭代学习。如果直接赋值作为估计值（self.alpha = 1）
    # 迭代次数会大大增加，如果考虑一定的学习率（alpha = 0.5），迭代次数会有显著的减少

    def policy_evaluation(self):
        t = 0
        max_dff = 0
        while 1:
            new_v = [0] * self.env.ncol * self.env.nrow
            t += 1
            for i in range(len(self.v)):
                for a in range(4):
                    new_q = 0
                    next_v = self.env.P[i][a]
                    for next_s in next_v:
                        (p, next_state, r, done) = next_s
                        new_q += p * (r + self.gamma * self.v[next_state]) if not done else p * r
                    new_v[i] += self.pi[i][a] * new_q
                    self.q[i][a] = new_q
            max_diff = (max([abs(new_v[tmp] - self.v[tmp]) for tmp in range(len(new_v))]))
            if max_diff < self.theta:
                print("the evaluation for policy is done after %d times" % t)
                break
            else:
                self.v = new_v

    # 在improvement中，我们同样不需要进行学习率的考量，因为我们直接贪心：将最优的动作价值函数的动作作为策略
    # self.pi = max(Q(s,a)) / (num(max(S,a)))
    def policy_improvement(self):
        for i in range(len(self.pi)):
            self.pi[i] = [1 / (self.q[i].count(max(self.q[i]))) if a_q == max(self.q[i]) else 0 for a_q in self.q[i]]
        print("the improvement is done")
        return self.pi

    def policy_iteration(self):

        for _ in range(100):
            self.cnt += 1
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  # 将列表进行深拷贝,方便接下来进行比较
            new_pi = self.policy_improvement()
            if old_pi == new_pi:
                print("the learning is done after %d iteration" % self.cnt)
                break


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]),
                  end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


if __name__ == "__main__":
    action_meaning = ['^', 'v', '<', '>']
    agent = PolicyIteration()
    agent.policy_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])
