import time
import gym
import pybullet as p
import peg_in_hole_gym
from peg_in_hole_gym.envs.base_env import TASK_LIST
from env import ArtForce
from test_env import TestForce
from tqdm import tqdm

from icecream import install
install()

TASK_LIST['art-force'] = ArtForce
TASK_LIST['test-force'] = TestForce

object_list = ["microwave", "toaster", "drawer", "cabinet", "cabinet2", "refrigerator"]

if __name__ == '__main__':
    env = gym.make('peg-in-hole-v0', client=p.GUI, task="art-force", task_num=1, offset = [2.,3.,0.],args=[object_list[0], True, True], is_test=True)
    # env = gym.make('peg-in-hole-v0', client=p.GUI, task="test-force", task_num=1, offset = [2.,3.,0.],args=[object_list[0], False, True], is_test=True)
    env.reset()
    # env.step([[1]])
    # while True:
    #     env.step([[1]])
    #     time.sleep(0.01)
    cnt = 8000
    for i in tqdm(range(cnt)):
        env.step([[1]])
        time.sleep(0.01)
    env.render()
