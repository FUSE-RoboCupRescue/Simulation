import rowan
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.wheely_generator import gen


class WheelyEnv(gym.Env):

    def __init__(self, step_update_rate: int=2, render: bool=False):
        super(WheelyEnv, self).__init__()

        self.discount_factor = .9999
        self.sim_timestep = .002
        self.wheely_body = "chassis"
        self.goal = "goal"
        self.max_steps = 2_000
        self.step_update_rate = self.sim_timestep * step_update_rate
        self.generated_xml_path, self.xml_info = gen(self.sim_timestep)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

    def get_obs(self) -> dict:

        body = self.model.body(self.wheely_body)
        rot = rowan.to_matrix(self.data.xquat[body.id])[:2].flatten()
        vel = self.data.cvel[body.id]
        obs = np.hstack((rot, vel))

        return obs
    
    def get_reward(self) -> float:

        body_wheely = self.model.body(self.wheely_body)
        body_goal = self.model.body(self.goal)
        p1_xy = self.data.xpos[body_wheely.id][:2]
        p2_xy = self.data.xpos[body_goal.id][:2]
        
        reward = -1. * np.linalg.norm( p2_xy - p1_xy )

        return reward
    
    def reset(self, seed: int=42) -> tuple:

        np.random.seed(seed)

        # Load mujoco scene
        self.model = mujoco.MjModel.from_xml_path(self.generated_xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0

        obs = self.get_obs()
        info = {}
        return obs, info
    
    def step(self, action: list[float]) -> tuple:
        
        # Check if actions are valid
        assert self.action_space.contains(action), "Invalid action for action_space!"

        # Flipper motors
        self.data.ctrl[-4:] = action[6:] * self.xml_info["flipper_max_torque"]

        cs = [
            "main_indices_l",
            "main_indices_r",
            "flipper_indices_0",
            "flipper_indices_1",
            "flipper_indices_2",
            "flipper_indices_3"
            ]
        
        # Track motors
        for i, c in enumerate(cs):
            self.data.ctrl[self.xml_info[c]] = action[i] * self.xml_info["track_max_torque"]

        # Update simulator
        loop_count = int(self.step_update_rate / self.sim_timestep)
        for _ in range(loop_count):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        # Return states
        obs = self.get_obs()
        reward = self.get_reward()
        terminated = self.current_step > self.max_steps
        truncated = self.current_step > self.max_steps
        info = {}
        return obs, reward, terminated, truncated, info
    

if __name__ == "__main__":

    env = WheelyEnv()
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
