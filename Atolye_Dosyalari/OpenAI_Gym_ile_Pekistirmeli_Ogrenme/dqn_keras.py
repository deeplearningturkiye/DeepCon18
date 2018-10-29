import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import losses

class policy():
    def __init__(self, action_space):
        self.nA = action_space

    def probs(self, action_values,epsilon):
        a_probs = np.ones(self.nA, dtype=float) * epsilon / self.nA
        best_action = np.argmax(action_values)
        a_probs[best_action] += (1 - epsilon)

        return a_probs


class DQN:
    def __init__(self, state_size, action_size, epsilon=0.5, epsilon_min=0.1, model_test=False, epsilon_decrease=False, epsilon_decay=0.95):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.learning_rate = 0.001
        self.epsilon_decrease = epsilon_decrease
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.policy = policy(action_space=self.action_size)
        if not model_test:
            self.model = Sequential()
            self._build_model()
        else:
            self.epsilon = 0
            print("For test purpose, the epsilon value was drawn to 0.")

    def _build_model(self):
        self.model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_size, activation='linear'))
        self.model.compile(loss=losses.mean_squared_error,
                           optimizer=optimizers.Adam(lr=self.learning_rate))

    def remember(self, state, action, reward, next_state, done):
        state = np.array([state])
        next_state = np.array([next_state])
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        state = np.array([state])  # vectorize for prediction
        q_values_for_actions = self.model.predict(state)
        a_prob = self.policy.probs(q_values_for_actions, self.epsilon)

        return np.random.choice(np.arange(len(a_prob)), p=a_prob)

    def learn_from_memory(self, batch_size):
        if len(self.memory)>batch_size:
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = (reward + self.gamma *
                              np.amax(self.model.predict(next_state)[0]))

                predicted_q_values = self.model.predict(state)  # [[predicted_q_value_of_action_1,predicted_q_value_of_action_2]]

                q_values_for_training = predicted_q_values
                q_values_for_training[0][action] = target       # new action q values(for training)

                self.model.fit(state, q_values_for_training, epochs=1, verbose=0)  #trainig with new q values
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def save_model(self, name):
        self.model.save_weights(name)

    def load_model(self, name):
        self.model.load_weights(name)


