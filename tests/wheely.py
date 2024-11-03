import mujoco_py
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.wheely_generator import gen


gen()
# Load the MuJoCo model
model = mujoco_py.load_model_from_path("../xml_files/wheely.xml")
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

for i in range(model.nsite):
    site_name = model.site_names[i]
    site_pos = model.site_pos[i]
    print(f"Site Name: {site_name}, Position: {site_pos}")

# Simulate for a few steps
time_stamp = 0
while True:
    
    # Example control input for the motors
    sim.data.ctrl[:-4] = 1.
    # if time_stamp % 300 == 0:
    #     strenght = 100
    #     sim.data.ctrl[-1] = np.random.random()*strenght*2 -strenght
    #     sim.data.ctrl[-2] = np.random.random()*strenght*2 -strenght
    #     sim.data.ctrl[-3] = np.random.random()*strenght*2 -strenght
    #     sim.data.ctrl[-4] = np.random.random()*strenght*2 -strenght
    # sim.data.ctrl[-1] = 0
    sim.data.ctrl[-2] = -10.
    # sim.data.ctrl[-3] = 0
    sim.data.ctrl[-4] = -10.
    print(sim.data.ctrl)
    sim.step()
    viewer.render()
    time_stamp += 1
