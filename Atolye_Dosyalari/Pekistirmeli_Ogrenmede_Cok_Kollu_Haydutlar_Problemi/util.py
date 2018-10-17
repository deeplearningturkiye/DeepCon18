import numpy as np
import matplotlib.pyplot as plt


class bandit:
    def __init__(self, k_arm=10, eps=0, step_size=0.1):
        self.k_arm = k_arm
        self.epsilon = eps
        self.step_size = step_size

        self.q_true = np.random.randn(self.k_arm)
        self.q_estimates = np.zeros(self.k_arm)

        self.best_arm = np.argmax(self.q_true)

    def action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k_arm)

        return np.argmax(self.q_estimates)

    def step(self):
        idx = self.action()

        reward = self.q_true[idx] + np.random.randn()

        self.q_estimates[idx] += self.step_size * (reward - self.q_estimates[idx])

        return reward

    def take_steps(self, count):
        for _ in np.arange(count):
            self.step()

        self.plot_current_estimates()

    def plot_current_estimates(self):
        plt.violinplot(positions=np.arange(self.k_arm), dataset=self.q_true + np.random.randn(100,10))
        plt.plot(self.q_estimates, color='red', marker='o', linestyle='', markersize=5)
        plt.show()
