import gym
env = gym.make('CartPole-v0')

env.reset()
rewards = []

episodes=10
max_time=100
for e in range(episodes):
    env.render()
    for t in range(max_time) :
        state, reward, done, info = env.step(env.action_space.sample()) # rastgele hareket et

        if done:
            rewards.append(t)
            env.reset()



print(rewards)
env.close()