import numpy as np
import matplotlib.pyplot as plt


class bandit:
    def __init__(self, eps=0, step_size=0., initial=0, variance=1):
        self.epsilon = eps
        self.step_size = step_size
        self.k_arm = 5
        self.variance = variance

        self.q_true = np.asarray([-0.25, 2, 1, -1.5, 3])
        self.q_estimates = np.zeros(self.k_arm) + initial

        self.actions_taken = np.zeros(self.k_arm)

    def action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k_arm)
        else:
            return np.argmax(self.q_estimates)

    def step(self):
        idx = self.action()

        self.actions_taken[idx] += 1

        reward = self.q_true[idx] + np.random.randn() * self.variance

        if self.step_size == 0:
            self.q_estimates[idx] = self.q_estimates[idx] + (1.0 / self.actions_taken[idx]) * (reward - self.q_estimates[idx])
        else:
            self.q_estimates[idx] = self.q_estimates[idx] + self.step_size * (reward - self.q_estimates[idx])

        return reward

    def change_q_true(self):
        self.q_true = np.asarray([1, -0.5, -2, 2, 0.25])

    def take_steps(self, count):
        for _ in np.arange(count):
            self.step()

        self.showdown()

    def plot(self):
        plt.violinplot(positions=np.arange(self.k_arm), dataset=self.q_true + np.random.randn(100,self.k_arm) * self.variance)
        plt.plot(self.q_estimates, color='red', marker='o', linestyle='', markersize=5)
        plt.show()

    def dump(self):
        for i in np.arange(self.k_arm):
            print('Arm #%u: %u times.' % (i, self.actions_taken[i]))

        print('Best Arm: #%u.' % (np.argmax(self.q_true)))

    def showdown(self):
        self.dump()
        self.plot()
