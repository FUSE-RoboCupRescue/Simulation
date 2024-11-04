from enviroments.mj_enviroments import WheelyEnv


env = WheelyEnv()
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
