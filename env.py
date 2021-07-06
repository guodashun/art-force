import os
import math
import random
import numpy as np
from PIL import Image

import pybullet as p
import pybullet_data
from peg_in_hole_gym.envs.meta_env import MetaEnv

from generation.generator_pybullet import SceneGenerator

class ArtForce(MetaEnv):
    # action_space=spaces.Box(np.array([-0.8,0,0,-math.pi,-math.pi,-math.pi]),np.array([0.8,0.8,0.8,math.pi,math.pi,math.pi]))
    # observation_space = spaces.Box(np.array([0,0,0,-math.pi/3*2]), np.array([1,1,1,0]))
    obj_list = ["microwave", "toaster", "drawer", "cabinet", "cabinet2", "refrigerator"]
    def __init__(self, client, offset, args=["microwave", True, True]):
        assert args[0] in self.obj_list, f"[x] {args[0]} is not in obj_list, valid object is {self.obj_list}"
        self.obj = args[0]
        self.mean_flag = args[1]
        self.only_left = args[2]
        super().__init__(client, offset)

    
    def _load_models(self):
        
        # load dataset
        self._load_articulated_object()

        # fix object
        self._fix_object()

        # add gripper
        self._add_gripper()
        # # test
        # print(f"[*] TEST FOR GRIPPER")
        # joint_num = self.p.getNumJoints(self.gripper_id)
        # for i in range(joint_num):
        #     joint_info =  self.p.getJointInfo(self.gripper_id, i)
        #     print("[*] Joint info:", joint_info)

    
    def _reset_internals(self):
        
        # reset(random) articulated object pose
        for i in self.axis_index:
            angle = random.random()*(-math.pi/18) - math.pi/18
            if self.obj == "refrigerator":
                angle = -angle
            self.p.resetJointState(self.obj_id, i, angle)
            print(f"[~] axis {i} has been reset")

        # reset virtual tcp pose

    def apply_action(self, action):
        # joint_num = self.p.getNumJoints(self.obj_id)
        # for i in range(joint_num):
        #     joint_info =  self.p.getJointState(self.obj_id, i)
        #     print("[*] Joint info:", i, joint_info)

        # make virtual end go arc

        # just test
        # self.p.resetBaseVelocity(self.gripper_id, [0.1,0.1,0.1])
        pass
    
    def get_info(self):
        # get the prediction of the axis's pose
        return [], 0, False, []

    
    def reset(self, hard_reset):
        super().reset(hard_reset=hard_reset)


    def _load_articulated_object(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        scene = SceneGenerator(self.p, root_dir=f'{dir_path}/assets')

        # print(f"[*] show me the self.obj: {self.obj}")
        obj,_,_ = scene.sample_obj(self.obj, self.mean_flag, self.only_left)
        xml = obj.xml
        fname=os.path.join(scene.savedir, f'{self.obj}.xml')
        scene.write_urdf(fname, xml)
        
        objId, _ = self.p.loadMJCF(fname)
        self.obj_id = objId

        # create normal texture image
        x, y = np.meshgrid(np.linspace(-1,1, 128), np.linspace(-1,1, 128))
        texture_img = (72*(np.stack([np.cos(16*x), np.cos(16*y), np.cos(16*(x+y))])+2)).astype(np.uint8).transpose(1,2,0)
        texture_img = Image.fromarray(texture_img)
        fname = 'normal_texture.png'
        texture_img.save(fname)
        textureId = self.p.loadTexture(fname)

        # create gaussian texture image
        SHAPE = (150,200)
        noise = np.random.normal(255./1,255./3,SHAPE)
        image_noise = Image.fromarray(noise)
        image_noise = image_noise.convert('RGB')
        fname2 = "gaussian_noise.png"
        image_noise.save(fname2)
        textureId2 = self.p.loadTexture(fname2)

        # apply texture to the object way: idea two
        # self.p.changeVisualShape(objId, -1, textureUniqueId=textureId2, rgbaColor=[1, 1, 1, 1], specularColor=[1, 1, 1, 1], physicsClientId=pb_client) #bottom 
        self.p.changeVisualShape(objId, 0, textureUniqueId=textureId2, rgbaColor=[1, 1, 1, 1], specularColor=[1, 1, 1, 1]) #left side
        self.p.changeVisualShape(objId, 1, textureUniqueId=textureId2, rgbaColor=[1, 1, 1, 1], specularColor=[1, 1, 1, 1]) #right side
        self.p.changeVisualShape(objId, 2, textureUniqueId=textureId2, rgbaColor=[1, 1, 1, 1], specularColor=[1, 1, 1, 1]) #top

    def _fix_object(self):
        # get object's l&w
        joint_num = self.p.getNumJoints(self.obj_id)
        bottom_link_pos = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0])
        l, w = -1,-1
        axis_index = []
        for i in range(joint_num):
            joint_info =  self.p.getJointInfo(self.obj_id, i)
            # print("[*] Joint info:", joint_info)
            if joint_info[12] == b'cabinet_left':
                l = abs(joint_info[14][1])
            elif joint_info[12] == b'cabinet_back':
                w = abs(joint_info[14][0])
            elif joint_info[1] == b'bottom_left_hinge':
                axis_index.append(i)
            elif self.obj in ["cabinet2", "refrigerator"] and joint_info[1] == b'bottom_right_hinge':
                axis_index.append(i)
        assert l!=-1 and w!=-1, "[x] Cannot find object h/w!"
        print(f"[~] object's l: {l}, w: {w}")
        assert axis_index, "[x] Cannot find any axis!"
        print(f"[~] axis_index: {axis_index}")
        self.axis_index = axis_index
        
        # add box to fix object
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # table_list = []
        for i in range(4):
            cube_id = self.p.loadURDF(
                "cube.urdf", 
                basePosition=bottom_link_pos+np.array([w*(1 if i > 1 else -1), l*(1 if i % 2 else -1), -0.02]),
                baseOrientation=self.p.getQuaternionFromEuler([0, 0, -math.pi]),
                globalScaling=0.02,
                useFixedBase=True,
                flags=0
            )
            self.p.createConstraint(
                cube_id,
                -1,
                self.obj_id,
                -1,
                self.p.JOINT_FIXED,
                [0,0,1],
                np.array([w*(1 if i > 1 else -1), l*(1 if i % 2 else -1),  0.02]),
                [0,0,0],
                # childFrameOrientation=[0,0,math.pi/2]
            )
            # print("[*] debug for offset", np.array([w*(1 if i > 1 else -1), l*(1 if i % 2 else -1),  -0.02]))
        print("[~] Constraint established.")  

    def _add_gripper(self):
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.gripper_id = self.p.loadSDF(
            f"gripper/wsg50_one_motor_gripper.sdf", 
            # basePosition=bottom_link_pos+np.array([w*(1 if i > 1 else -1), l*(1 if i % 2 else -1), -0.02]),
            # baseOrientation=self.p.getQuaternionFromEuler([0, 0, -math.pi]),
            # globalScaling=0.02,
            # useFixedBase=True,
            # flags=0
        )[0]


# if __name__ == '__main__':
#     env = ArtForce()

    