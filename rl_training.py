from stable_baselines3 import SAC

from enviroments.mj_enviroments import WheelyEnv


env = WheelyEnv()

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1, log_interval=1)
model.save("./data/models/wheely_ai")

del model

model = SAC.load("./data/models/wheely_ai")

obs, info = env.reset()
for _ in range(3):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
