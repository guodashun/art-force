import time
import gym
import pybullet as p
import peg_in_hole_gym
from peg_in_hole_gym.envs.base_env import TASK_LIST
from env import ArtForce

TASK_LIST['art-force'] = ArtForce

if __name__ == '__main__':
    env = gym.make('peg-in-hole-v0', client=p.GUI, task="art-force", task_num=1, offset = [2.,3.,0.],args=[], is_test=True)
    env.reset()
    while True:
        env.step([[1]])
        time.sleep(0.1)