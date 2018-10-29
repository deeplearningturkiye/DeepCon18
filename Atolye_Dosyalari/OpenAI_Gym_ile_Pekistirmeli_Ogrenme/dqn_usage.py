from dqn_keras import DQN
import gym

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQN(state_size, action_size, epsilon=0.9, epsilon_decrease=True, epsilon_min=0.1)
done = False
batch_size = 32
max_time = 250
episodes = 20
scores = []
for e in range(episodes):
    state = env.reset()
    for time in range(max_time):
        env.render()
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -10  # if done we have to punish the agent with -10
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            scores.append(time)
            print("episode: {}/{}, score: {}"
                  .format(e+1, episodes, time))
            break
        agent.learn_from_memory(batch_size)
print(scores)
agent.save_model("model.h5")

scores = []
agent = DQN(state_size, action_size, epsilon=0.9, epsilon_decrease=True, epsilon_min=0.2)
agent.load_model("model.h5")
done = False
for e in range(episodes):
    state = env.reset()
    for time in range(max_time):
        env.render()
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -10  # if done we have to punish the agent with -10
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            scores.append(time)
            print("episode: {}/{}, score: {}"
                  .format(e+1, episodes, time))
            break
        agent.learn_from_memory(batch_size)
print(scores)
agent.save_model("model.h5")

scores = []
model = agent.get_model()


agent = DQN(state_size, action_size, model_test=True)
agent.set_model(model)
# veya agent.load_model("model.h5")

for e in range(episodes):
    state = env.reset()
    for time in range(max_time):
        env.render()
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            scores.append(time)
            print("episode: {}/{}, score: {}"
                  .format(e+1, episodes, time))
            break
print(scores)


from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import losses
learning_rate = 0.001
yeni_model = model = Sequential()
model.add(Dense(48, input_dim=env.observation_space.shape[0], activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(action_size, activation='linear'))
model.compile(loss=losses.mean_squared_error,
              optimizer=optimizers.Adam(lr=learning_rate))

scores = []
agent = DQN(state_size, action_size, epsilon=0.9, epsilon_decrease=True, epsilon_min=0.2)
agent.set_model(model)
done = False
for e in range(episodes):
    state = env.reset()
    for time in range(max_time):
        env.render()
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        if done:
            reward = -10  # if done we have to punish the agent with -10
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            scores.append(time)
            print("episode: {}/{}, score: {}"
                  .format(e+1, episodes, time))
            break
        agent.learn_from_memory(batch_size)
print(scores)
