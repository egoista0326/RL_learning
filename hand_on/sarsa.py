import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标

    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
            else:
                reward = 0
        return next_state, reward, done

    def reset(self):  # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x


class Sarsa:
    def __init__(self, env, gamma=0.9, alpha=0.1, n_action=4, epsilon=0.9, epoch=20, episode=5):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q = np.zeros([self.env.nrow * self.env.ncol, n_action])
        self.epoch = epoch
        self.episode = episode
        self.n_action = n_action
        self.return_list = []

    def take_action(self, state):
        if np.random.random() > self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.q[state])
        if action is None:
            print("action_next is None")
        return action

    def best_action(self, state):
        Q_max = np.max(self.q[state, :])
        best_compare = [1 if self.q[state, i] == Q_max else 0 for i in range(self.q.shape[1])]
        return best_compare

    def learn(self, state, action, reward, state_next, action_next):
        if state_next == -1:
            self.q[state, action] += self.alpha * (reward - self.q[state, action])
        else:
            # print(state, action, state_next, action_next)
            self.q[state, action] += self.alpha * (
                    reward + self.gamma * self.q[state_next, action_next] - self.q[state, action])

    def train_per(self):
        state = self.env.reset()
        done = 0
        action = self.take_action(state)
        reward_total = 0
        while not done:
            state_next, reward, done = self.env.step(action)
            reward_total += reward
            assert (0 <= action < 4)
            assert (0 <= state_next < self.env.nrow * self.env.ncol)
            if not done:
                action_next = self.take_action(state_next)
                self.learn(state, action, reward, state_next, action_next)
                state = state_next
                action = action_next
            else:
                self.learn(state, action, reward, -1, 0)
        return reward_total


    def train(self):
        for e in range(self.epoch):
            with tqdm(range(self.episode), desc="Training Progress") as epoch_pbar:  # 外层进度条
                for i in epoch_pbar:
                    j = 0
                    reward_total = 0
                    while j < self.episode:
                        j += 1
                        reward = self.train_per()
                        # print(reward)
                        reward_total += reward
                        # print(reward_total)
                    # 更新外层进度条的信息
                    epoch_pbar.update(1)  # 确保每个epoch结束时更新进度条
                    epoch_pbar.set_postfix({
                        'epoch': '%d' % (e + 1),
                        'reward': '%.3f' % (reward_total/ j )  # 避免除以零
                    })
                    self.return_list.append(reward)


if __name__ == "__main__":
    env = CliffWalkingEnv(12, 4)
    sarsa = Sarsa(env=env)
    sarsa.train()
    # print(sarsa.q)
    print("训练完成！")

    episodes_list = list(range(len(sarsa.return_list)))
    plt.plot(episodes_list, sarsa.return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Sarsa on Cliff Walking')
    plt.show()