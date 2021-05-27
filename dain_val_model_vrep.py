# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import matplotlib
matplotlib.use("TkAgg")
import rclpy
from policy_translation.srv import NetworkPT
from pyrep import PyRep
from pyrep.objects.vision_sensor import VisionSensor
from utils.voice import Voice
import sys
import select
import numpy as np
import cv2
import sensor_msgs.msg
import math
import pickle
import glob
import json
import csv
import os.path
import os
import matplotlib.pyplot as plt
from PIL import Image
from semantic_parser import semantic_parser
import time
from dain_object_detector import show_image
import tensorflow as tf

# Default robot position. You don't need to change this
DEFAULT_UR5_JOINTS  = [105.0, -30.0, 120.0, 90.0, 60.0, 90.0]
# Evaluate headless or not
HEADLESS            = False
# This is a debug variable... 
USE_SHAPE_SIZE      = True
# Run on the test data, or start the simulator in manual mode 
# (manual mode will allow you to generate environments and type in your own commands)
RUN_ON_TEST_DATA    = True
# How many of the 100 test-data do you want to test?
NUM_TESTED_DATA     = 1
# Where to find the normailization?
NORM_PATH           = "../GDrive/normalization_v2.pkl"
# Where to find the VRep scene file. This has to be an absolute path. 
# VREP_SCENE          = "../GDrive/NeurIPS2020.ttt"
# VREP_SCENE          = "../GDrive/testscene.ttt"
VREP_SCENE          = "../GDrive/testscene2.ttt"
VREP_SCENE          = os.getcwd() + "/" + VREP_SCENE
CUP_ID_TO_NAME      = {21: 'red cup', 22: 'green cup', 23: 'blue cup'}
BIN_ID_TO_NAME      = {16: 'yellow dish', 17: 'red dish', 18: 'green dish', 19: 'blue dish', 20: 'pink dish'} # HACK
SIM_CUP_TO_FRCNN    = {1:21, 2:22, 3:23, 4:21, 5:22, 6:23}
SIM_BIN_TO_FRCNN    = {1:16, 2:17, 3:18, 4:19, 5:20}

class Simulator(object):
    def __init__(self, args=None):
        rclpy.init(args=args)
        self.node       = rclpy.create_node("VRepSimTest")
        self.srv_prx_nn = self.node.create_client(NetworkPT, "/network")

        self.node.get_logger().info("Service not available, waiting...")
        while not self.srv_prx_nn.wait_for_service(timeout_sec=1.0):
            pass
        self.node.get_logger().info("... service found!")

        self.pyrep = PyRep()
        self.pyrep.launch(VREP_SCENE, headless=HEADLESS)
        self.camera = VisionSensor("kinect_rgb_full")
        self.pyrep.start()

        self.trajectory    = None
        self.global_step   = 0
        self.normalization = pickle.load(open(NORM_PATH, mode="rb"), encoding="latin1")
        self.voice         = Voice(load=False)  

        self.shape_size_replacement = {}
        self.shape_size_replacement["58z29D2omoZ_2.json"] = "spill everything into the large curved dish"
        self.shape_size_replacement["P1VOZ4zk4NW_2.json"] = "fill a lot into the small square basin"
        self.shape_size_replacement["KOVJZ4Npy4G_2.json"] = "fill a small amount into the big round pot"
        self.shape_size_replacement["wjqQmB74rnr_2.json"] = "pour all of it into the large square basin"
        self.shape_size_replacement["LgVK8qXGowA_2.json"] = "fill a little into the big round bowl"
        self.shape_size_replacement["JZ90qm46ooP_2.json"] = "fill everything into the biggest rectangular bowl"

        self.subtasks = []
        self.subtask_idx = 0
    
    def loadNlpCSV(self, path):
        self.nlp_dict = {}
        with open(path, "r") as fh:
            csvreader = csv.reader(fh, delimiter=",")
            for line in csvreader:
                if line[1] != "":
                    self.nlp_dict[line[0]+"_1.json"] = line[1]
                    self.nlp_dict[line[0]+"_2.json"] = line[2]

    def shutdown(self):
        self.pyrep.stop()
        self.pyrep.shutdown()

    def _getCameraImage(self):
        rgb_obs = self.camera.capture_rgb()
        rgb_obs = (np.asarray(rgb_obs) * 255).astype(dtype=np.uint8)
        rgb_obs = np.flip(rgb_obs, (2))
        return rgb_obs
    
    def _getSimulatorState(self):
        _, s, _, _ = self.pyrep.script_call(function_name_at_script_name="getState@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")
        return s
    
    def _stopRobotMovement(self):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="stopRobotMovement@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")
    
    def _getRobotState(self):
        _, s, _, _ = self.pyrep.script_call(function_name_at_script_name="getState@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")
        state = np.take(np.asarray(s), indices=[0,1,2,3,4,5,30], axis=0)  
        return state.tolist()

    def _setRobotJoints(self, joints):
            result = self.pyrep.script_call(function_name_at_script_name="setRobotJoints@control_script",
                                            script_handle_or_type=1,
                                            ints=(), floats=joints, strings=(), bytes="")

    def _setJointVelocityFromTarget(self, joints):
        _, s, _, _ = self.pyrep.script_call(function_name_at_script_name="setJointVelocityFromTarget@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=joints, strings=(), bytes="")

    def _setJointVelocityFromTarget_Direct(self, joints):
        _, s, _, _ = self.pyrep.script_call(function_name_at_script_name="setJointVelocityFromTarget_Direct@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=joints, strings=(), bytes="")
    
    def _dropBall(self, b_id):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="dropBall@control_script",
                                        script_handle_or_type=1,
                                        ints=(b_id,), floats=(), strings=(), bytes="")
    
    def _evalPouring(self):
        i, _, _, _ = self.pyrep.script_call(function_name_at_script_name="evalPouring@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")
        return i
    
    # def _evalPlacing(self):
    #     _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="evalPlacing@control_script",
    #                                     script_handle_or_type=1,
    #                                     ints=(), floats=(), strings=(), bytes="")

    def _getClosestBin(self):
        i, _, _, _ = self.pyrep.script_call(function_name_at_script_name="getClosestBin@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")
        # if i[0] >= 0:
        #     return True
        # return False
        return i
    
    def _graspedObject(self):
        i, _, _, _ = self.pyrep.script_call(function_name_at_script_name="graspedObject@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")
        # if i[0] >= 0:
        #     return True
        # return False
        return i
    
    def _setRobotInitial(self, joints):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="setRobotJoints@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=joints, strings=(), bytes="")
    
    def _graspClosestContainer(self):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="graspClosestContainer@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")

    def _randomizeLight(self):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="randomizeLight@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")

    def _releaseObject(self):
        _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="releaseObject@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")

    def _resetEnvironment(self):
        self.pyrep.stop()
        self.pyrep.start()
    
    def _createEnvironment(self, ints, floats):
        result = self.pyrep.script_call(
            function_name_at_script_name="generateScene@control_script",
            script_handle_or_type=1,
            ints=ints, 
            floats=floats, 
            strings=(), 
            bytes=""
        )

    def _getClosesObject(self):
        oid, dist, _, _ = self.pyrep.script_call(function_name_at_script_name="getClosesObject@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")
        return oid, dist

    def dtype_with_channels_to_cvtype2(self, dtype, n_channels):
        numpy_type_to_cvtype = {'uint8': '8U', 'int8': '8S', 'uint16': '16U',
                                     'int16': '16S', 'int32': '32S', 'float32': '32F',
                                     'float64': '64F'}
        numpy_type_to_cvtype.update(dict((v, k) for (k, v) in numpy_type_to_cvtype.items()))
        return '%sC%d' % (numpy_type_to_cvtype[dtype.name], n_channels)

    def cv2_to_imgmsg(self, cvim, encoding="passthrough"):
        if not isinstance(cvim, (np.ndarray, np.generic)):
            raise TypeError('Your input type is not a numpy array')
        img_msg = sensor_msgs.msg.Image()
        img_msg.height = cvim.shape[0]
        img_msg.width = cvim.shape[1]
        if len(cvim.shape) < 3:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, 1)
        else:
            cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, cvim.shape[2])
        if encoding == 'passthrough':
            img_msg.encoding = cv_type
        else:
            img_msg.encoding = encoding
            # Verify that the supplied encoding is compatible with the type of the OpenCV image
            if self.cvtype_to_name[self.encoding_to_cvtype2(encoding)] != cv_type:
                raise CvBridgeError('encoding specified as %s, but image has incompatible type %s'
                                    % (encoding, cv_type))
        if cvim.dtype.byteorder == '>':
            img_msg.is_bigendian = True
        img_msg.data.frombytes(cvim.tobytes())
        img_msg.step = len(img_msg.data) // img_msg.height

        return img_msg
    
    def predictTrajectory(self, voice, state, cnt):
        norm         = np.take(self.normalization["values"], indices=[0,1,2,3,4,5,30], axis=1)
        image        = self._getCameraImage()

        robot_state    = state
        try:
            robot_state[6] = self.last_gripper
        except:
            robot_state[6] = 0.0

        req          = NetworkPT.Request()
        req.image    = self.cv2_to_imgmsg(image[:,:,::-1])
        req.language = self._generalizeVoice(voice)
        req.robot    = self.normalize([robot_state], norm[0,:], norm[1,:]).flatten().tolist()
        req.reset    = cnt == 1
        req.plot     = False

        future       = self.srv_prx_nn.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        try:
            result = future.result()
        except Exception as e:
            self.node.get_logger().info('Service call failed %r' % (e,))
            return False
        
        trajectory = np.asarray(result.trajectory).reshape(-1, 7)
        trajectory = self.restoreValues(trajectory, norm[0,:], norm[1,:])
        phase      = float(result.phase)
        features   = result.features
        features   = np.reshape(features, (6,5))
        attn       = result.attn
        return trajectory, phase, features, attn
    
    def normalize(self, value, v_min, v_max):
        if type(value) == list:
            value = np.asarray(value)
        if (value.shape[1] != v_min.shape[0] or v_min.shape[0] != v_max.shape[0] or 
            len(value.shape) != 2 or len(v_min.shape) != 1 or len(v_max.shape) != 1):
            raise ArrayDimensionMismatch()
        value = np.copy(value)
        v_min = np.tile(np.expand_dims(v_min, 0), [value.shape[0], 1])
        v_max = np.tile(np.expand_dims(v_max, 0), [value.shape[0], 1])
        value = (value - v_min) / (v_max - v_min)
        return value
    
    def restoreValues(self, value, v_min, v_max):
        if (value.shape[1] != v_min.shape[0] or v_min.shape[0] != v_max.shape[0] or 
            len(value.shape) != 2 or len(v_min.shape) != 1 or len(v_max.shape) != 1):
            print("Array dimensions are not matching!")

        value = np.copy(value)
        v_min = np.tile(np.expand_dims(v_min, 0), [value.shape[0], 1])
        v_max = np.tile(np.expand_dims(v_max, 0), [value.shape[0], 1])
        value = value * (v_max - v_min) + v_min
        return value

    def _generalizeVoice(self, voice):
        return voice

    def _mapObjectIDs(self, oid):
        if oid == 127: # red cup
            return 21
        elif oid == 128: # red cup
            return 21
        elif oid == 129: # green cup
            return 22
        elif oid == 130: # green cup
            return 22
        elif oid == 126: # blue cup
            return 23
        elif oid == 131: # blue cup
            return 23

        elif oid == 117: # yellow bin
            return 16
        elif oid == 116: # red bin
            return 17
        elif oid == 115: # green bin
            return 18
        elif oid == 114: # blue bin (not used)
            return 19
        elif oid == 113: # pink bin (not used)
            return 20

        elif oid == -1: # nothing is grabbed
            return 0
        
        else:
            print('unidentified object in mapOIDs')
            print(oid)
            return 0
    
    def _correctGrab(self, command, grabbed_cup):
        if 'cup' not in command:
            return False
        if grabbed_cup == 'NONE':
            return False
        return grabbed_cup in command or not any([i in command for i in ['red','green','blue']])
    
    def _correctPlace(self, command, placed_bin):
        if 'dish' not in command:
            return False
        if placed_bin == 'NONE':
            return False
        return placed_bin in command or not any([i in command for i in ['red','green','blue']])
    
    def _evalObjectDetection(self, ground_truth, detected):
        '''
        e.g.
        ground truth: [21,17,17,0,0,0] -- one red cup, two red bins
        detected:     [21,17,0,0,0,0] -- one red cup, one red bin
        '''
        n_bins = ground_truth[0]
        n_cups = ground_truth[1]
        bins = [SIM_BIN_TO_FRCNN[i] for i in ground_truth[2:2+n_bins]]
        cups = [SIM_CUP_TO_FRCNN[i] for i in ground_truth[2+n_bins:]]
        gt = bins + cups + [0] * (6-len(bins+cups))
        
        return {} # TODO

    def _evalAttention(self, ground_truth, attn):
        '''
        e.g.
        ground truth: [1,0,0,0,0,0] -- command involving "red cup"
        attn:         [.5,.9,0,0,0,0] -- misdetected red bin as red cup
        '''
        pass

    def _getTargetPosition(self, data):
        state  = self._getSimulatorState()
        tcp    = state[12:14]
        target = data["target/id"]
        tp     = data["target/type"]
        if tp == "cup":
            cups   = data["ints"][2+data["ints"][0]:]
            t_id   = [i for i in range(data["ints"][1]) if cups[i] == target][0] + data["ints"][0]
            t_pos  = data["floats"][t_id*3:t_id*3+2]
        else:
            bowls  = data["ints"][2:2+data["ints"][0]]
            t_id   = [i for i in range(data["ints"][0]) if bowls[i] == target][0]
            t_pos  = data["floats"][t_id*3:t_id*3+2]

        dist   = np.sqrt( np.power(tcp[0] - t_pos[0], 2) + np.power(tcp[1] - t_pos[1], 2) )

        closest       = list(self._getClosesObject())
        closest[0][0] = self._mapObjectIDs(closest[0][0])
        closest[0][1] = self._mapObjectIDs(closest[0][1])
        result = {}
        result["target"]     = t_pos
        result["tid"]        = target
        result["tid/actual"] = closest
        result["current"]    = tcp
        result["distance"]   = dist
        return result

    def _maybeDropBall(self, state):
        res = 0
        if state[5] > 3.0:
            self._dropBall(1)
            # print('release object')
            # self._releaseObject()
            res = 1
        if state[5] > 3.0 and self.last_rotation > state[5]:
            self._dropBall(2)
            # print('release object')
            # self._releaseObject()
            res = 2
        self.last_rotation = state[5]
        return res
    
    def _maybeRelease(self, state):
        res = 0
        if state[5] > 2.0:
            print('releasing')
            self._releaseObject()
            res = 2
        self.last_rotation = state[5]
        return res
    
    def _getTaskCommandInfo(self, command, task_type, subtasks):
        data = {}
        data["command"]   = command
        data["task_type"] = task_type
        data["subtasks"]  = subtasks
        data['subtasks_info'] = self._getSubtasksInfo(subtasks)
        return data
    
    def _getSubtasksInfo(self, subtasks):
        subtasks_info = {}
        subtasks_info['pick_total'] = sum([1 for i in subtasks if 'pick' in i])
        subtasks_info['place_total'] = sum([1 for i in subtasks if 'pour' in i]) # HACK
        subtasks_info['cup_total'] = sum([1 for i in subtasks if 'cup' in i])
        subtasks_info['bin_total'] = sum([1 for i in subtasks if 'dish' in i]) # HACK
        subtasks_info['cup_by_type'] = {}
        subtasks_info['bin_by_type'] = {}
        for cup_type in CUP_ID_TO_NAME.values():
            subtasks_info['cup_by_type'][cup_type] = sum([1 for i in subtasks if cup_type in i])
        for bin_type in BIN_ID_TO_NAME.values():
            subtasks_info['bin_by_type'][bin_type] = sum([1 for i in subtasks if bin_type in i])
        
        return subtasks_info
        

    def _getLanguateInformation(self, voice, phs):
        def _quantity(voice):
            res = 0
            for word in self.voice.synonyms["little"]:
                if voice.find(word) >= 0:
                    res = 1
            for word in self.voice.synonyms["much"]:
                if voice.find(word) >= 0:
                    res = 2
            return res

        def _difficulty(voice):
            if phs == 2:
                voice  = " ".join(voice.split()[4:])
            shapes = self.voice.synonyms["round"] + self.voice.synonyms["square"]
            colors = self.voice.synonyms["small"] + self.voice.synonyms["large"]  
            sizes  = self.voice.synonyms["red"]   + self.voice.synonyms["green"] + self.voice.synonyms["blue"] + self.voice.synonyms["yellow"] + self.voice.synonyms["pink"]

            shapes_used = 0
            for word in shapes:
                if voice.find(word) >= 0:
                    shapes_used = 1
            colors_used = 0
            for word in colors:
                if voice.find(word) >= 0:
                    colors_used = 1
            sizes_used = 0
            for word in sizes:
                if voice.find(word) >= 0:
                    sizes_used = 1
            return shapes_used + colors_used + sizes_used

        data = {}
        data["original"] = voice
        data["features"] = _difficulty(voice)
        data["quantity"] = _quantity(voice)
        return data
    
    def evaluate(self, files, task_type):
        summary = {
            'pick_successful': 0,
            'pick_total': 0,
            'place_successful': 0,
            'place_total': 0
        }
        val_result = {}

        for fid, fn in enumerate(files):
            print("{} Run {}/{}".format(task_type, fid, len(files)))
            eval_data = {}
            """
            eval_data:
                'language': dict output of _getTaskCommandInfo (contains 'command', 'task_type', 'subtasks', 'subtasks_info')
                'trajectory': list of 7 DOF robot joints at each step
                'object_detection': dict output of _evalObjectDetection
            """
            with open(fn, "r") as fh:
                data = json.load(fh)

            # initial env setup
            self._resetEnvironment()
            self._createEnvironment(data["ints"], data["floats"])
            self.pyrep.step()
            _, _, features, _ = self.predictTrajectory("", self._getRobotState(), 1)
            feature_ids = [int(i) for i in features.T[0]]
            subtasks = semantic_parser(data["voice"], feature_ids)

            eval_data["language"] = self._getTaskCommandInfo(data["voice"], task_type, subtasks)
            eval_data["trajectory"] = []
            eval_data["object_detection"] = self._evalObjectDetection(data["ints"], features)
            summary['pick_total'] += eval_data['language']['subtasks_info']['pick_total']
            summary['place_total'] += eval_data['language']['subtasks_info']['place_total']

            subtask_idx = 0
            rm_voice = subtasks[subtask_idx]
            cnt   = 0
            phase = 0.0
            self.last_gripper = 0.0
            th = 1.0
            while phase < th:
                cnt += 1
                state = self._getRobotState()
                tf_trajectory, phase, features, attn = self.predictTrajectory(rm_voice, state, cnt)
                r_state    = tf_trajectory[-1,:]
                eval_data["trajectory"].append(r_state.tolist())
                # r_state[6] = r_state[6]
                self.last_gripper = r_state[6]
                self._setJointVelocityFromTarget(r_state)
                self.pyrep.step()
                
                if phase > 0.95:
                    # check if robot grabbed the correct cup
                    if 'pick' in rm_voice:
                        grasped_cup = self._graspedObject()[0]
                        if grasped_cup > 0:
                            grabbed_cup = CUP_ID_TO_NAME[self._mapObjectIDs(grasped_cup)]
                            # print('subtask was',rm_voice)
                            # print('robot grabbed', grabbed_cup)
                            if self._correctGrab(rm_voice, grabbed_cup):
                                print('successfully grabbed the right cup')
                                summary['pick_successful'] += 1
                
                    # check if robot placed cup in the correct bin
                    if 'pour' in rm_voice:
                        placed_bin = self._getClosestBin()[0]
                        if placed_bin > 0:
                            placed_bin = BIN_ID_TO_NAME[self._mapObjectIDs(placed_bin)]
                            if self._correctPlace(rm_voice, placed_bin):
                                print('successfully placed in the right bin')
                                summary['place_successful'] += 1

                        self._releaseObject()
                        self.resetRobotArm()
                    self._stopRobotMovement()

                    subtask_idx += 1
                    # if subtasks remain, keep going
                    if len(subtasks) > subtask_idx:
                        # print('moving onto next subtask')
                        rm_voice = subtasks[subtask_idx]
                        cnt = 0
                        phase = 0.0
                        print("Running Task: " + rm_voice)
                    else:
                        # if self.subtasks[-1].startswith('pour'):
                        #     self.resetRobotArm()
                        rm_voice = ""
                        subtasks = []
                        subtask_idx = 0
                        phase = 1.0

            val_result[data["name"]] = eval_data
            
        return summary, val_result

    # def valPhase1(self, files, feedback=True):
    #     successfull = 0
    #     val_data    = {}
    #     nn_trajectory  = []
    #     ro_trajectory  = []
    #     for fid, fn in enumerate(files):
    #         print("Phase 1 Run {}/{}".format(fid, len(files)))
    #         eval_data = {}
    #         with open(fn + "1.json", "r") as fh:
    #             data = json.load(fh)            

    #         gt_trajectory = np.asarray(data["trajectory"])
    #         self._resetEnvironment()
    #         self._createEnvironment(data["ints"], data["floats"])
    #         self._setRobotInitial(gt_trajectory[0,:])
    #         self.pyrep.step()

    #         eval_data["language"] = self._getLanguateInformation(data["voice"], 1)
    #         eval_data["trajectory"] = {"gt": [], "state": []}
    #         eval_data["trajectory"]["gt"] = gt_trajectory.tolist()

    #         cnt   = 0
    #         phase = 0.0
    #         self.last_gripper = 0.0
    #         th = 1.0
    #         while phase < th and cnt < int(gt_trajectory.shape[0] * 1.5):
    #             state = self._getRobotState() if feedback else gt_trajectory[-1 if cnt >= gt_trajectory.shape[0] else cnt,:]
    #             cnt += 1
    #             tf_trajectory, phase, features, attn = self.predictTrajectory(data["voice"], state, cnt)
    #             r_state    = tf_trajectory[-1,:]
    #             eval_data["trajectory"]["state"].append(r_state.tolist())
    #             r_state[6] = r_state[6] 
    #             nn_trajectory.append(r_state)
    #             ro_trajectory.append(self._getRobotState())
    #             self.last_gripper = r_state[6]
    #             self._setJointVelocityFromTarget(r_state)
    #             self.pyrep.step()
    #             if r_state[6] > 0.5 and "locations" not in eval_data.keys():
    #                 eval_data["locations"] = self._getTargetPosition(data)

    #         eval_data["success"] = False
    #         if self._graspedObject():
    #             eval_data["success"] = True
    #             successfull += 1
    #         val_data[data["name"]] = eval_data
            
    #     return successfull, val_data
    
    # def valPhase2(self, files, feedback=True):
    #     successfull         = 0
    #     val_data            = {}
    #     for fid, fn in enumerate(files):
    #         print("Phase 2 Run {}/{}".format(fid, len(files)))
    #         eval_data = {}
    #         fpath     = fn + "2.json"
    #         filename  = os.path.basename(fpath)
    #         with open(fpath, "r") as fh:
    #             data = json.load(fh)
    #         gt_trajectory = np.asarray(data["trajectory"])
            
    #         if USE_SHAPE_SIZE and filename in self.shape_size_replacement.keys():
    #             data["voice"] = self.shape_size_replacement[filename]

    #         self._resetEnvironment()
    #         self._createEnvironment(data["ints"], data["floats"])
    #         self._setRobotInitial(gt_trajectory[0,:])
    #         self.pyrep.step()
    #         self._graspClosestContainer()
    #         self.pyrep.step()

    #         self.last_gripper  = 1.0
    #         self.last_rotation = 0.0

    #         eval_data["language"] = self._getLanguateInformation(data["voice"], 2)
    #         eval_data["trajectory"] = {"gt": [], "state": []}
    #         eval_data["trajectory"]["gt"] = gt_trajectory.tolist()

    #         cnt   = 0
    #         phase = 0.0
    #         th = 1.0
    #         while phase < th and cnt < int(gt_trajectory.shape[0] * 1.5):
    #             state = self._getRobotState() if feedback else gt_trajectory[-1 if cnt >= gt_trajectory.shape[0] else cnt,:]
    #             cnt += 1
    #             tf_trajectory, phase, _, attn = self.predictTrajectory(data["voice"], self._getRobotState(), cnt)
    #             r_state              = tf_trajectory[-1,:]
    #             # r_state[5] = 0.0 # TODO last angle of rotation?
    #             eval_data["trajectory"]["state"].append(r_state.tolist())
    #             r_state[6]           = r_state[6]
    #             self._setJointVelocityFromTarget(r_state)
    #             self.last_gripper = r_state[6]
    #             # dropped = self._maybeDropBall(r_state)
    #             dropped = self._maybeRelease(r_state)
    #             if dropped == 1 and "locations" not in eval_data.keys():
    #                 eval_data["locations"] = self._getTargetPosition(data)
    #             self.pyrep.step()

    #         presult                 = self._evalPouring()
    #         eval_percentage         = np.sum(presult) / float(len(presult))
    #         eval_data["ball_array"] = presult
    #         if eval_percentage > 0.5:
    #             successfull += 1
    #             eval_data["success"] = True
    #         else:
    #             eval_data["success"] = False
    #         val_data[data["name"]] = eval_data
            
    #     return successfull, val_data

    def evalDirect(self, runs):
        # files = glob.glob("../GDrive/dain/testdata/*_1.json")
        sort_files = [f"sort_{i}.json" for i in range(1, 2)]
        kit_files = [f"kit_{i}.json" for i in range(1, 2)]
        # self.node.get_logger().info("Using data directory with {} files".format(len(sort_files)+len(kit_files)))
        # files = files[:runs]
        # files = [f[:-6] for f in files]
        self.node.get_logger().info("Running validation on {} files".format(len(sort_files)+len(kit_files)))

        data = {}
        sort_summary, sort_result = self.evaluate(sort_files, 'sorting')
        kit_summary, kit_result   = self.evaluate(kit_files, 'kitting')
        data["sorting"]           = sort_result
        data["kitting"]           = kit_result

        self.node.get_logger().info("Testing Sorting (Pick): {}/{} ({:.1f}%)".format(sort_summary['pick_successful'],sort_summary['pick_total'], 100.0*sort_summary['pick_successful']/sort_summary['pick_total']))
        self.node.get_logger().info("Testing Sorting (Place): {}/{} ({:.1f}%)".format(sort_summary['place_successful'],sort_summary['place_total'], 100.0*sort_summary['place_successful']/sort_summary['place_total']))
        self.node.get_logger().info("Testing Kitting (Pick): {}/{} ({:.1f}%)".format(kit_summary['pick_successful'],kit_summary['pick_total'], 100.0*kit_summary['pick_successful']/kit_summary['pick_total']))
        self.node.get_logger().info("Testing Kitting (Place): {}/{} ({:.1f}%)".format(kit_summary['place_successful'],kit_summary['place_total'], 100.0*kit_summary['place_successful']/kit_summary['place_total']))
        # self.node.get_logger().info("Testing Kitting: {}/{} ({:.1f}%)".format(s_p2,  runs, 100.0 * float(s_p2)/float(runs)))

        # TODO
        # p1_names = data["phase_1"].keys()
        # # p2_names = data["phase_2"].keys()
        # # names = [n for n in p1_names if n in p2_names]
        # names = [n for n in p1_names]
        # c_p2  = 0
        # for n in names:
        #     if data["phase_1"][n]["success"]:
        #         c_p2  += 1

        # self.node.get_logger().info("Whole Task: {}/{} ({:.1f}%)".format(c_p2,  len(names), 100.0 * float(c_p2)  / float(len(names))))

        with open("val_result.json", "w") as fh:
            json.dump(data, fh)
        
    
    def _generateEnvironment(self):
        def genPosition(prev):
            px = 0
            py = 0
            done = False        
            while not done:
                done = True        
                px = np.random.uniform(-0.9, 0.35)
                py = np.random.uniform(-0.9, 0.35)
                dist = np.sqrt(px**2 + py**2)
                if dist < 0.5 or dist > 0.9:
                    done = False
                for o in prev:
                    if np.sqrt((px - o[0])**2 + (py - o[1])**2) < 0.25:
                        done = False
                if px > 0 and py > 0:
                    done = False
                angle = -45
                r_px  = px * np.cos(np.deg2rad(angle)) + py * np.sin(np.deg2rad(angle))
                r_py  = py * np.cos(np.deg2rad(angle)) - px * np.sin(np.deg2rad(angle))
                if r_py > 0.075:
                    done = False
            return [px, py]
        self._setRobotJoints(np.deg2rad(DEFAULT_UR5_JOINTS))

        # Max 6 objects per scene
        ncups  = np.random.randint(1,5)
        nbowls = np.random.randint(1,7-ncups)
        bowls  = np.random.choice(5, size=nbowls, replace=False) + 1
        cups   = np.random.choice(6, size=ncups, replace=False) + 1
        ints   = [nbowls, ncups] + bowls.tolist() + cups.tolist()
        floats = []

        prev   = []
        for i in range(nbowls + ncups):
            prev.append(genPosition(prev))
            floats += prev[-1]
            if i < nbowls:
                floats += [np.random.uniform(-math.pi/4.0,  math.pi/4.0)]
            else:
                floats += [0.0]
            # floats += [0.0]

        self._createEnvironment(ints, floats)
        self.node.get_logger().info("Created new environment")
        print('ints',ints)
        print('floats',floats)
        print('rstate', self._getRobotState())
        print('sim state', self._getSimulatorState())
        return ints, floats
    
    def _generateSetEnvironment(self, idx):
        def genPosition(prev):
            px = 0
            py = 0
            done = False        
            while not done:
                done = True        
                px = np.random.uniform(-0.9, 0.35)
                py = np.random.uniform(-0.9, 0.35)
                dist = np.sqrt(px**2 + py**2)
                if dist < 0.5 or dist > 0.9:
                    done = False
                for o in prev:
                    if np.sqrt((px - o[0])**2 + (py - o[1])**2) < 0.25:
                        done = False
                if px > 0 and py > 0:
                    done = False
                angle = -45
                r_px  = px * np.cos(np.deg2rad(angle)) + py * np.sin(np.deg2rad(angle))
                r_py  = py * np.cos(np.deg2rad(angle)) - px * np.sin(np.deg2rad(angle))
                if r_py > 0.075:
                    done = False
            return [px, py]
        self._setRobotJoints(np.deg2rad(DEFAULT_UR5_JOINTS))

        # Sim 0: default
        if idx == "0":
            ints = [5, 6, 1,2,3,4,5, 1,2,3,4,5,6]
            floats = []
            bin_pos = [0, -1, 0]
            cup_pos = [-1, 0, 0]
            for i in range(5):
                floats += bin_pos
                bin_pos[0] += 0.25
                bin_pos[1] += 0.25
            for i in range(6):
                floats += cup_pos
                cup_pos[0] += 0.1
                cup_pos[1] += 0.1

        # Sim 1: one cup and one bin
        # "put the cup in the bin"
        if idx == "1":
            ints = [1,1,1,1]
            floats = []
            prev   = []
            for i in range(2):
                prev.append(genPosition(prev))
                floats += prev[-1]
                if i < ints[0]:
                    floats += [np.random.uniform(-math.pi/4.0,  math.pi/4.0)]
                else:
                    floats += [0.0]

        # Sim 2: two cups with different colors and one bin
        # "put the red cup in the bin"
        elif idx == "2":
            ints = [1,2,2,1,3]
            floats = []
            prev   = []
            for i in range(3):
                prev.append(genPosition(prev))
                floats += prev[-1]
                if i < ints[0]:
                    floats += [np.random.uniform(-math.pi/4.0,  math.pi/4.0)]
                else:
                    floats += [0.0]

            floats = [-0.6706283735164196, -0.367048866651563, 0.5677358902060654, -0.38444069922215474, -0.5240614358568401, 0.0, -0.005800754313754042, -0.6434291140615598, 0.0]
            # floats = [-0.4841615916276253, -0.3829838314703341, -0.05021171096973509, -0.7201993483166631, -0.12716774121561158, 0.0, -0.34769231510157994, -0.6539479616451864, 0.0]


        # Sim 2: two cups with same colors and one bin
        # "put the red cup in the bin"
        elif idx == "3":
            ints = [1,2,2, 1,2]
            floats = []
            prev   = []
            for i in range(3):
                prev.append(genPosition(prev))
                floats += prev[-1]
                if i < ints[0]:
                    floats += [np.random.uniform(-math.pi/4.0,  math.pi/4.0)]
                else:
                    floats += [0.0]

            # floats = [-0.6706283735164196, -0.367048866651563, 0.5677358902060654, -0.38444069922215474, -0.5240614358568401, 0.0, -0.005800754313754042, -0.5434291140615598, 0.0]
            floats = [-0.4841615916276253, -0.3829838314703341, -0.05021171096973509, -0.7201993483166631, -0.00716774121561158, 0.0, -0.34769231510157994, -0.6539479616451864, 0.0]



        # Sim 3: several cups and one bin
        # # "put all the red cups in the bin"
        # elif idx == "3":
        #     ints = [1,3,1,1,3,4]
        #     floats = []
        #     prev   = []
        #     for i in range(4):
        #         prev.append(genPosition(prev))
        #         floats += prev[-1]
        #         if i < ints[0]:
        #             floats += [np.random.uniform(-math.pi/4.0,  math.pi/4.0)]
        #         else:
        #             floats += [0.0]

        # Sim 4: several cups and several bins
        # "put all the red cups in the blue bin"
        elif idx == "4":
            ints = [2,3,1,2,1,3,4]
            floats = []
            prev   = []
            for i in range(5):
                prev.append(genPosition(prev))
                floats += prev[-1]
                if i < ints[0]:
                    floats += [np.random.uniform(-math.pi/4.0,  math.pi/4.0)]
                else:
                    floats += [0.0]
            # floats = [-0.6195587892326153, -0.27958541999686426, -0.5158047614218091, 0.23493611762502653, -0.6759613027633606, 0.005197504389217067, -0.48792603175619886, -0.5265215716212565, 0.0, -0.37264993848291743, -0.8466030247075923, 0.0, -0.5827080589449817, -0.033880902918536515, 0.0]
            floats = [-0.8486552682267293, 0.03691380392628727, 0.454911799489471, 
                      -0.6829666818647918, -0.34188501757644943, -0.2013484777087384, 
                      -0.2271148259848611, -0.6246002557744265, 0.0, 
                      -0.39214640134532, 0.3494459948857429, 0.0, 
                      0.05249601319387964, -0.5748357248031177, 0.0]
            print(ints)
            print(floats)
        
        self._createEnvironment(ints, floats)
        self.node.get_logger().info("Created new set environment {}".format(idx))
        return ints, floats

    def simplifyVoice(self, voice):
        simple = []
        for word in voice.split(" "):
            if word in self.voice.basewords.keys():
                simple.append(self.voice.basewords[word])
        return " ".join(simple)

    def parseInput(self, d_in):
        if d_in == "q":
            return False
        if d_in == "g":
            self.rm_voice     = ""
            self.last_gripper = 0.0
            self._generateEnvironment()
        if d_in in ("0", "1", "2", "3", "4", "5"):
            self.rm_voice     = ""
            self.last_gripper = 0.0
            self._generateSetEnvironment(d_in)
        if d_in == "r":
            self.rm_voice     = ""
            self.last_gripper = 0.0
            self.node.get_logger().info("Resetting robot")
            self._resetEnvironment()
            self.subtasks = []
            self.subtask_idx = 0
        if d_in == 'd':
            self.releaseRobotGrip()
        if d_in == 's':
            print('current robot state', self._getRobotState())
            q_prime = np.append(np.deg2rad(DEFAULT_UR5_JOINTS),[0.0])
            q = self._getRobotState()
            error = np.linalg.norm(q - q_prime)
            print('error: ',error)
        if d_in == 'z':
            self.resetRobotArm()
        elif d_in.startswith("t "):
            # self.rm_voice = d_in[2:]
            # get the scene objects
            _, _, features, attn = self.predictTrajectory("", self._getRobotState(), 1)
            feature_ids = [int(i) for i in features.T[0]]
            self.subtasks = semantic_parser(d_in[2:], feature_ids)
            self.subtask_idx = 0
            self.rm_voice = self.subtasks[self.subtask_idx]
            self.cnt      = 0
            print("Running Task: " + self.rm_voice)
        elif self.rm_voice != "" and  d_in == "":
            # run robot
            self.cnt += 1
            tf_trajectory, phase, features, attn = self.predictTrajectory(self.rm_voice, self._getRobotState(), self.cnt)
            r_state              = tf_trajectory[-1,:]
            # r_state = 6x robot joint position (j1, j2, j3, j4, j5, j6) + gripper position
            # hack: no rotation
            # r_state[5] = 1.5
            # print('r_state',r_state)
            # self._graspedObject()

            self.last_gripper    = r_state[6]
            # print('gripper:', self.last_gripper)
            self._setJointVelocityFromTarget(r_state)
            # self._maybeDropBall(r_state)
            # released = self._maybeRelease(r_state)
            # hack for releasing object:
            # if self.rm_voice.startswith('pour') and phase > 0.3:
            #     print('releasing')
            #     self._releaseObject()
            #     phase = 0.95

            if phase > 0.98:
                self.node.get_logger().info("Finished running trajectory with " + str(self.cnt) + " steps")
                # self._releaseObject()
                if 'pour' in self.subtasks[self.subtask_idx]:
                    self._releaseObject()
                    self.resetRobotArm()
                self._stopRobotMovement()
                # self.rm_voice = ""

                self.subtask_idx += 1
                # if subtasks remain, keep going
                if len(self.subtasks) > self.subtask_idx:
                    print('moving onto next subtask')
                    self.rm_voice = self.subtasks[self.subtask_idx]
                    self.cnt = 0
                    print("Running Task: " + self.rm_voice)
                else:
                    if self.subtasks[-1].startswith('pour'):
                        self.resetRobotArm()
                    self.rm_voice = ""
                    self.subtasks = []
                    self.subtask_idx = 0


        return True
    
    def releaseRobotGrip(self):
        self._releaseObject()
        self.last_gripper = 0.0
    
    def resetRobotArm(self):
        self._stopRobotMovement()
        self.rm_voice     = ""
        # self._releaseObject()
        q_prime = np.append(np.deg2rad(DEFAULT_UR5_JOINTS),[0.0])
        q = self._getRobotState()
        error = np.linalg.norm(q - q_prime)
        steps = max(2, int(error*30))
        while error > 0.001:
            joint_trj = np.linspace(q, q_prime, num=steps, endpoint=True, axis=0)
            for trj in joint_trj:
                self._setJointVelocityFromTarget(trj)
                self.pyrep.step()
            q = self._getRobotState()
            error = np.linalg.norm(q - q_prime)
            steps = max(2, int(error*30))
        self._stopRobotMovement()
        self._setRobotJoints(np.deg2rad(DEFAULT_UR5_JOINTS))
        # self._setJointVelocityFromTarget(self._getRobotState())
        print('robot arm position reset')
    
    def runManually(self):
        self.rm_voice = ""
        run  = True
        while run:
            self.pyrep.step()
            if select.select([sys.stdin,],[],[],0.0)[0]:
                line = sys.stdin.readline()
                run = self.parseInput(line[:-1])
            else:
                self.parseInput("")    
        print("Shutting down...")
  
if __name__ == "__main__":
    sim = Simulator()
    if RUN_ON_TEST_DATA:
        sim.evalDirect(runs=NUM_TESTED_DATA)
        # for i,img in enumerate(imgs):
            # cv2.imwrite(f"img_{i}.png", img)
            # show_image(img)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
    else:
        sim.runManually()
    sim.shutdown()
    