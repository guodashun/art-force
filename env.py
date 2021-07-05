import os
import math
import numpy as np
from PIL import Image

import pybullet as p
import pybullet_data
from peg_in_hole_gym.envs.meta_env import MetaEnv

from generation.generator_pybullet import SceneGenerator

class ArtForce(MetaEnv):
    # action_space=spaces.Box(np.array([-0.8,0,0,-math.pi,-math.pi,-math.pi]),np.array([0.8,0.8,0.8,math.pi,math.pi,math.pi]))
    # observation_space = spaces.Box(np.array([0,0,0,-math.pi/3*2]), np.array([1,1,1,0]))
    def __init__(self, client, offset, args=[]):

        super().__init__(client, offset)

    
    def _load_models(self):

        # init table
        # table_pos = np.array([0.0,0,-1.4]) + self.offset
        # table_orn = np.array([0,0,0])
        # self.init_table(table_pos, table_orn)
        
        # load dataset
        scene = SceneGenerator(self.p, root_dir='/home/luckky/luckky/art-force/test')

        obj,_,_ = scene.sample_obj("microwave", True, True)
        xml = obj.xml
        fname=os.path.join(scene.savedir, 'scene.xml')
        scene.write_urdf(fname, xml)
        # objId = self.p.loadMJCF(fname)
        obj_id = self._visual_object(fname)
        self.obj_id = obj_id

        # fix object to table
        joint_num = self.p.getNumJoints(obj_id)
        # bottom_link_id = -1
        bottom_link_pos = np.array(self.p.getBasePositionAndOrientation(obj_id)[0])
        l, w = -1,-1
        for i in range(joint_num):
            joint_info =  self.p.getJointInfo(obj_id, i)
            print("[*] Joint info:", joint_info)
            if joint_info[12] == b'cabinet_left':
                l = abs(joint_info[14][1])
            elif joint_info[12] == b'cabinet_back':
                w = abs(joint_info[14][0])
        
        assert l!=-1 and w!=-1, "Cannot find object h/w!"
        print(f"[~] object's l: {l}, w: {w}")
        # add box
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
                obj_id,
                -1,
                self.p.JOINT_FIXED,
                [0,0,1],
                np.array([w*(1 if i > 1 else -1), l*(1 if i % 2 else -1),  0.02]),
                [0,0,0],
                # childFrameOrientation=[0,0,math.pi/2]
            )
            # print("[*] debug for offset", np.array([w*(1 if i > 1 else -1), l*(1 if i % 2 else -1),  -0.02]))
            

        
        # assert bottom_link_id != -1, "Cannot find link \'cabinet_back\'!"
        # print(f"[~] Find bottom link id: {bottom_link_id}")
        # bottom_link_pos = np.array(self.p.getLinkState(obj_id, bottom_link_id)[0])
        
        print("[*] bottom_frame_pos:", bottom_link_pos)
        # self.p.createConstraint(
        #     self.table_id, 
        #     -1, 
        #     obj_id, 
        #     -1, 
        #     self.p.JOINT_FIXED,
        #     [0,0,0],
        #     bottom_link_pos+[0,0,1],
        #     [0,0,0],
        #     parentFrameOrientation=[0,0,math.pi]
        # )
        # self.p.createConstraint(
        #     self.table_id, 
        #     -1, 
        #     obj_id, 
        #     bottom_link_id, 
        #     self.p.JOINT_FIXED,
        #     [0.1,0,0],
        #     bottom_link_pos+[0,0.1,1],
        #     [0,0,0],
        #     childFrameOrientation=[0,0,math.pi]
        # )
        print("[~] Constraint established.")

        
    
    def _reset_internals(self):
        pass
        
        # reset(random) articulated object pose

        # reset virtual tcp pose

    def _visual_object(self, filename):
        objId, _ = self.p.loadMJCF(filename)

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


        # apply texture to the object way: idea one
        # planeVis = self.p.createVisualShape(shapeType=self.p.GEOM_MESH,
        #                        fileName=filename,
        #                        rgbaColor=[168 / 255.0, 164 / 255.0, 92 / 255.0, 1.0], 
        #                        specularColor=[0.5, 0.5, 0.5],
        #                        physicsClientId=pb_client)

        # self.p.changeVisualShape(planeVis,
        #                     -1,
        #                     textureUniqueId=textureId,
        #                     rgbaColor=[1, 1, 1, 1],
        #                     specularColor=[1, 1, 1, 1],
        #                     physicsClientId=pb_client)

        # apply texture to the object way: idea two
        # self.p.changeVisualShape(objId, -1, textureUniqueId=textureId2, rgbaColor=[1, 1, 1, 1], specularColor=[1, 1, 1, 1], physicsClientId=pb_client) #bottom 
        self.p.changeVisualShape(objId, 0, textureUniqueId=textureId2, rgbaColor=[1, 1, 1, 1], specularColor=[1, 1, 1, 1]) #left side
        self.p.changeVisualShape(objId, 1, textureUniqueId=textureId2, rgbaColor=[1, 1, 1, 1], specularColor=[1, 1, 1, 1]) #right side
        self.p.changeVisualShape(objId, 2, textureUniqueId=textureId2, rgbaColor=[1, 1, 1, 1], specularColor=[1, 1, 1, 1]) #top
        return objId


    def apply_action(self, action):
        # make virtual end go arc
        pass
    
    def get_info(self):
        # get the prediction of the axis's pose
        return [], 0, False, []

    
    def reset(self, hard_reset):
        super().reset(hard_reset=hard_reset)


# if __name__ == '__main__':
#     env = ArtForce()

    