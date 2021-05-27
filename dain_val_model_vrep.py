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
from tensorflow import sigmoid

# Default robot position. You don't need to change this
DEFAULT_UR5_JOINTS  = [105.0, -30.0, 120.0, 90.0, 60.0, 90.0]
# Evaluate headless or not
HEADLESS            = False
# Run on the test data, or start the simulator in manual mode 
# (manual mode will allow you to generate environments and type in your own commands)
RUN_ON_TEST_DATA    = False
# How many of the 100 test-data do you want to test?
NUM_TESTED_DATA     = 1
# Where to find the normailization?
NORM_PATH           = "../GDrive/normalization_v2.pkl"
# Where to find the VRep scene file. This has to be an absolute path. 
# VREP_SCENE          = "../GDrive/NeurIPS2020.ttt"
# VREP_SCENE          = "../GDrive/testscene.ttt"
VREP_SCENE          = "../GDrive/testscene2.ttt"
VREP_SCENE          = os.getcwd() + "/" + VREP_SCENE
# Data collection vars
COLLECT_DATA        = True
NUM_SAMPLES         = 100
DATA_PATH           = "./eval_data/"
# CUP_ID_TO_NAME      = {21: 'red cup', 22: 'green cup', 23: 'blue cup'}
# BIN_ID_TO_NAME      = {16: 'yellow dish', 17: 'red dish', 18: 'green dish', 19: 'blue dish', 20: 'pink dish'}
# SIM_CUP_TO_FRCNN    = {1:21, 2:22, 3:23, 4:21, 5:22, 6:23}
# SIM_BIN_TO_FRCNN    = {1:16, 2:17, 3:18, 4:19, 5:20}
CUP_ID_TO_NAME      = {21: 'red cup', 23: 'blue cup'} # only red and blue cups are used
BIN_ID_TO_NAME      = {16: 'yellow dish', 17: 'red dish', 18: 'green dish'} # HACK, also only yellow red green bins are used
SIM_CUP_TO_FRCNN    = {1:21, 3:23, 4:21, 6:23}
SIM_BIN_TO_FRCNN    = {1:16, 2:17, 3:18}

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
    
    # def _getSimulatorState(self):
    #     _, s, _, _ = self.pyrep.script_call(function_name_at_script_name="getState@control_script",
    #                                     script_handle_or_type=1,
    #                                     ints=(), floats=(), strings=(), bytes="")
    #     return s
    
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

    # def _setJointVelocityFromTarget_Direct(self, joints):
    #     _, s, _, _ = self.pyrep.script_call(function_name_at_script_name="setJointVelocityFromTarget_Direct@control_script",
    #                                     script_handle_or_type=1,
    #                                     ints=(), floats=joints, strings=(), bytes="")

    def _getClosestBin(self):
        i, _, _, _ = self.pyrep.script_call(function_name_at_script_name="getClosestBin@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")
        return i
    
    def _graspedObject(self):
        i, _, _, _ = self.pyrep.script_call(function_name_at_script_name="graspedObject@control_script",
                                        script_handle_or_type=1,
                                        ints=(), floats=(), strings=(), bytes="")
        return i
    
    # def _setRobotInitial(self, joints):
    #     _, _, _, _ = self.pyrep.script_call(function_name_at_script_name="setRobotJoints@control_script",
    #                                     script_handle_or_type=1,
    #                                     ints=(), floats=joints, strings=(), bytes="")

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
        elif oid == 129: # green cup (not used)
            return 22
        elif oid == 130: # green cup (not used)
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
    
    def _pickScore(self, command, grabbed_cup):
        # nothing is grabbed
        if grabbed_cup == -1:
            return 0
        grabbed_cup = CUP_ID_TO_NAME[self._mapObjectIDs(grabbed_cup)]
        return int(grabbed_cup in command or not any([i in command for i in ['red','green','blue']]))
    
    def _placeScore(self, command, placed_bin):
        # not near a bin
        if placed_bin == -1:
            return 0
        placed_bin = BIN_ID_TO_NAME[self._mapObjectIDs(placed_bin)]
        return int(placed_bin in command or not any([i in command for i in ['yellow','red','green']]))
    
    def _evalObjectDetection(self, ground_truth, detected):
        '''
        detection = {
            'ground_truth': [0,0,0,17,17,21],         -- one red cup, two red bins
            'detected_objects': [0,0,0,0,17,21]       -- one red cup, one red bin detected
        }
        '''
        detection = {}

        n_bins = ground_truth[0]
        n_cups = ground_truth[1]
        bins = [SIM_BIN_TO_FRCNN[i] for i in ground_truth[2:2+n_bins]]
        cups = [SIM_CUP_TO_FRCNN[i] for i in ground_truth[2+n_bins:]]
        ground_truth = sorted(bins + cups + [0] * (6-len(bins+cups)))
        detection['ground_truth'] = ground_truth
        detection['detected_objects'] = sorted(detected)

        return detection

    # def _evalAttention(self, subtasks):
    #     '''
    #     attention = [
    #         {
    #             'keyword': 'red cup',                 -- command involving "red cup"
    #             'ground_truth': [0,0,0,0,21,21],      -- two red cups
    #             'attended_objects': [0,0,0,0,0,21]    -- one red cup attended to
    #         }
    #     ]
    #     '''
    #     attention = []

    #     for subtask in subtasks:
    #         subtask_attn_result = {'keyword': self._findKeyword(subtask)}
    #         _, _, features, attn = self.predictTrajectory(subtask, self._getRobotState(), 1)
    #         feature_ids = [int(i) for i in features.T[0]]
    #         ground_truth = self._getAttentionGT(subtask, feature_ids)
            
    #         filter_1 = [idx for idx,i in enumerate(attn) if i>=(max(attn)-min(attn))*0.9+min(attn) and i>0]
    #         # filter_2 = np.argwhere(np.array(attn)>=max(0,int(max(attn)))).flatten().tolist()
    #         attended_objects = np.array(feature_ids)[filter_1].tolist()
    #         ground_truth = sorted(ground_truth + [0] * (6-len(ground_truth)))
    #         attended_objects = sorted(attended_objects + [0] * (6-len(attended_objects)))
    #         subtask_attn_result['ground_truth'] = ground_truth
    #         subtask_attn_result['attended_objects'] = attended_objects

    #         attention.append(subtask_attn_result)

    #     return attention

    def _evalAttention(self, subtask, features, attn):
        '''
        subtask_attention = {
            'keyword': 'red cup',                 -- command involving "red cup"
            'ground_truth': [0,0,0,0,21,21],      -- two red cups
            'attended_objects': [0,0,0,0,0,21]    -- one red cup attended to
        }
        '''
        
        subtask_attention = {'keyword': self._findKeyword(subtask)}
        # _, _, features, attn = self.predictTrajectory(subtask, self._getRobotState(), 1)
        feature_ids = [int(i) for i in features.T[0]]
        ground_truth = self._getAttentionGT(subtask, feature_ids)
        
        filter_1 = [idx for idx,i in enumerate(attn) if i>=(max(attn)-min(attn))*0.95+min(attn) and i>0]
        # filter_2 = np.argwhere(np.array(attn)>=max(0,int(max(attn)))).flatten().tolist()
        attended_objects = np.array(feature_ids)[filter_1].tolist()
        ground_truth = sorted(ground_truth + [0] * (6-len(ground_truth)))
        attended_objects = sorted(attended_objects + [0] * (6-len(attended_objects)))
        subtask_attention['ground_truth'] = ground_truth
        subtask_attention['attended_objects'] = attended_objects

        return subtask_attention
    
    def _getAttentionGT(self, command, feature_ids):
        # pick task
        if 'cup' in command:
            # generic pick task "pick up the cup"
            if not any([i in command for i in ['red','blue']]):
                return [i for i in feature_ids if i in list(CUP_ID_TO_NAME.keys())]
            # specific pick task "pick up the red cup"
            else:
                cup_id = 21 if 'red' in command else 23
                return [i for i in feature_ids if i == cup_id]

        # place task
        elif 'dish' in command: # HACK
            # generic place task "pour it in the dish"
            if not any([i in command for i in ['yellow','red','green']]):
                return [i for i in feature_ids if i in list(BIN_ID_TO_NAME.keys())]
            # specific pick task "pour it in the yellow dish"
            else:
                bin_id = 16 if 'yellow' in command else 17 if 'red' in command else 18
                return [i for i in feature_ids if i == bin_id]
        
        return []
    
    def _findKeyword(self, subtask):
        for keyword in list(CUP_ID_TO_NAME.values()) + list(BIN_ID_TO_NAME.values()) + ['cup', 'dish']:
            if keyword in subtask:
                return keyword
        return ''

    def _getLanguageInfo(self, command, eval_type, subtasks):
        data = {}
        data["command"]   = command
        data["eval_type"] = eval_type
        data["subtasks"]  = subtasks
        data['task_summary'] = self._getTaskSummary(subtasks)
        return data
    
    def _getTaskSummary(self, subtasks):
        task_summary = {}
        task_summary['pick_total'] = sum([1 for i in subtasks if 'pick' in i])
        task_summary['place_total'] = sum([1 for i in subtasks if 'pour' in i]) # HACK
        task_summary['cup_types'] = {}
        task_summary['bin_types'] = {}
        for cup_type in CUP_ID_TO_NAME.values():
            task_summary['cup_types'][cup_type] = sum([1 for i in subtasks if cup_type in i])
        task_summary['cup_types']['unspecified'] = task_summary['pick_total'] - sum(task_summary['cup_types'].values())
        for bin_type in BIN_ID_TO_NAME.values():
            task_summary['bin_types'][bin_type] = sum([1 for i in subtasks if bin_type in i])
        task_summary['bin_types']['unspecified'] = task_summary['place_total'] - sum(task_summary['bin_types'].values())
        
        return task_summary
    
    def evaluate(self, files, eval_type):
        eval_result = {}

        for fid, fn in enumerate(files):
            print("{} Run {}/{}".format(eval_type, fid+1, len(files)))
            eval_data = {}
            """
            eval_data:
                'language': dict output of _getLanguageInfo (contains 'command', 'eval_type', 'subtasks', 'task_summary')
                'object_detection': dict output of _evalObjectDetection (contains 'ground_truth', 'detected_objects')
                'attention': list of _evalAttention outputs (contains 'keyword', 'ground_truth', 'attended_objects' for each subtask)
                'control': dict of pick and place success rates (contains 'pick_success', 'pick_total', 'place_success', 'place_total')
                # 'trajectory': list of 7 DOF robot joints at each step
            """
            with open(fn, "r") as fh:
                data = json.load(fh)

            # initial env setup
            self._resetEnvironment()
            self._createEnvironment(data["ints"], data["floats"])
            self.last_gripper = 0.0
            self.pyrep.step()
            print(f"Executing {eval_type} command: ", data["task"])

            # generate subtasks based on task scene
            _, _, features, _ = self.predictTrajectory("", self._getRobotState(), 1)
            feature_ids = [int(i) for i in features.T[0]]
            subtasks = semantic_parser(data["task"], feature_ids)

            eval_data["language"] = self._getLanguageInfo(data["task"], eval_type, subtasks)
            eval_data["object_detection"] = self._evalObjectDetection(data["ints"], feature_ids)
            eval_data["attention"] = []
            eval_data["control"] = {
                'pick_success': 0,
                'pick_total': eval_data['language']['task_summary']['pick_total'],
                'place_success': 0,
                'place_total': eval_data['language']['task_summary']['place_total']
            }
            # eval_data["trajectory"] = []  # Not collecting trajectory to save space

            subtask_idx = 0
            cnt   = 0
            
            while len(subtasks) > subtask_idx:
                # progress simulation
                cnt += 1
                state = self._getRobotState()
                rm_voice = subtasks[subtask_idx]
                tf_trajectory, subtask_phase, features, attn = self.predictTrajectory(rm_voice, state, cnt)
                r_state = tf_trajectory[-1,:]
                # eval_data["trajectory"].append(r_state.tolist())
                self.last_gripper = r_state[6]
                self._setJointVelocityFromTarget(r_state)
                self.pyrep.step()

                # evaluate attention
                if len(eval_data["attention"]) <= subtask_idx:
                    eval_data["attention"].append(self._evalAttention(rm_voice, features, attn))
                
                # current subtask complete
                if subtask_phase > 0.95:
                    # check if robot grabbed the correct cup
                    if 'pick' in rm_voice:
                        eval_data["control"]['pick_success'] += self._pickScore(rm_voice, self._graspedObject()[0])
                
                    # check if robot placed cup in the correct bin
                    elif 'pour' in rm_voice:
                        eval_data["control"]['place_success'] += self._placeScore(rm_voice, self._getClosestBin()[0]) 
                        self._releaseObject()
                        self.resetRobotArm()

                    self._stopRobotMovement()
                    subtask_idx += 1
                    cnt = 0

            eval_result[data["name"]] = eval_data
            
        return eval_result

    def evalDirect(self, runs):
        # files = glob.glob("../GDrive/dain/testdata/*_1.json")
        sort_files = [f"{DATA_PATH}sorting_{i}.json" for i in range(2)]
        kit_files = [f"{DATA_PATH}kitting_{i}.json" for i in range(2)]
        # self.node.get_logger().info("Using data directory with {} files".format(len(sort_files)+len(kit_files)))
        # files = files[:runs]
        # files = [f[:-6] for f in files]
        self.node.get_logger().info("Running evaluation on {} files".format(len(sort_files)+len(kit_files)))

        data = {}
        sort_result     = self.evaluate(sort_files, 'sorting')
        kit_result      = self.evaluate(kit_files, 'kitting')
        data["sorting"] = sort_result
        data["kitting"] = kit_result

        # self.node.get_logger().info("Testing Sorting (Pick): {}/{} ({:.1f}%)".format(sort_summary['pick_successful'],sort_summary['pick_total'], 100.0*sort_summary['pick_successful']/sort_summary['pick_total']))
        # self.node.get_logger().info("Testing Sorting (Place): {}/{} ({:.1f}%)".format(sort_summary['place_successful'],sort_summary['place_total'], 100.0*sort_summary['place_successful']/sort_summary['place_total']))
        # self.node.get_logger().info("Testing Kitting (Pick): {}/{} ({:.1f}%)".format(kit_summary['pick_successful'],kit_summary['pick_total'], 100.0*kit_summary['pick_successful']/kit_summary['pick_total']))
        # self.node.get_logger().info("Testing Kitting (Place): {}/{} ({:.1f}%)".format(kit_summary['place_successful'],kit_summary['place_total'], 100.0*kit_summary['place_successful']/kit_summary['place_total']))

        # TODO log stuff
        # p1_names = data["phase_1"].keys()
        # # p2_names = data["phase_2"].keys()
        # # names = [n for n in p1_names if n in p2_names]
        # names = [n for n in p1_names]
        # c_p2  = 0
        # for n in names:
        #     if data["phase_1"][n]["success"]:
        #         c_p2  += 1

        # self.node.get_logger().info("Whole Task: {}/{} ({:.1f}%)".format(c_p2,  len(names), 100.0 * float(c_p2)  / float(len(names))))
        self.node.get_logger().info("Finished Running Evaluation!")

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
        nbowls = np.random.randint(1,4) # max 3 bins
        ncups = np.random.randint(1, min(5, 7-nbowls)) # max 4 cups
        bowls = np.random.choice([1,2,3], size=nbowls, replace=False)
        cups = np.random.choice([1,3,4,6], size=ncups, replace=False)

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

        self._createEnvironment(ints, floats)
        self.node.get_logger().info("Created new environment")
        # print('ints',ints)
        # print('floats',floats)
        # print('rstate', self._getRobotState())
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

        # Sim 3: two cups with same colors and one bin
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
            _, _, features, _ = self.predictTrajectory("", self._getRobotState(), 1)
            feature_ids = [int(i) for i in features.T[0]]
            self.subtasks = semantic_parser(d_in[2:], feature_ids)
            if self.subtasks == []:
                print('ERROR: command is malformed')
                return True
            self.subtask_idx = 0
            self.rm_voice = self.subtasks[self.subtask_idx]
            self.cnt      = 0
            print("Running Task: " + self.rm_voice)
        elif self.rm_voice != "" and  d_in == "":
            # run robot
            self.cnt += 1
            tf_trajectory, phase, features, attn = self.predictTrajectory(self.rm_voice, self._getRobotState(), self.cnt)
            r_state                              = tf_trajectory[-1,:]
            # r_state = 6x robot joint position (j1, j2, j3, j4, j5, j6) + gripper position

            self.last_gripper    = r_state[6]
            # print('gripper:', self.last_gripper)
            self._setJointVelocityFromTarget(r_state)

            if phase > 0.98:
                self.node.get_logger().info("Finished running trajectory with " + str(self.cnt) + " steps")
                # self._releaseObject()
                if 'pour' in self.subtasks[self.subtask_idx]:
                    self._releaseObject()
                    self.resetRobotArm()
                self._stopRobotMovement()
                # print('features', [int(i) for i in features.T[0]])
                # print('attn', attn)
                # # print('sigmoid', sigmoid(attn))
                # print('filtered', [idx for idx,i in enumerate(attn) if i>=(max(attn)-min(attn))*0.9+min(attn) and i>0])
                # print('filter 2', np.argwhere(np.array(attn)>=max(0,int(max(attn)))).flatten().tolist())
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
        # print('robot arm position reset')
    
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
    
    def collectData(self, num_samples, eval_type):
        for i in range(num_samples):
            sample_data = {}
            print(f"generating env for {eval_type}_{i}")
            ints, floats = self._generateEnvironment()
            sample_data["name"] = f"{eval_type}_{i}"
            sample_data["ints"] = ints
            sample_data["floats"] = floats
            sample_data["task"] = self._generateTaskCommand(ints, floats, eval_type)

            with open(DATA_PATH+sample_data['name']+'.json', 'w') as f:
                json.dump(sample_data, f)


    def _generateTaskCommand(self, ints, floats, task_type):
        n_bins = ints[0]
        n_cups = ints[1]
        bin_colors = [BIN_ID_TO_NAME[SIM_BIN_TO_FRCNN[i]][:-5] for i in ints[2:2+n_bins]]
        cup_colors = [CUP_ID_TO_NAME[SIM_CUP_TO_FRCNN[i]][:-4] for i in ints[2+n_bins:]]

        # "put all the ... cups in the ... dish"
        if task_type == 'sorting':
            task = "put all the {cup_color} cups in the {bin_color} dish"
            
            cup_color_choices = []
            if n_cups <= 2:
                cup_color_choices.append('')
            for cup_color in ['red', 'blue']:
                if cup_colors.count(cup_color) > 0:
                    cup_color_choices.append(cup_color)

            bin_color_choices = []
            if n_bins == 1:
                bin_color_choices.append('')
            for bin_color in ['yellow', 'red', 'green']:
                if bin_colors.count(bin_color) > 0:
                    bin_color_choices.append(bin_color)
            descs = {
                'cup_color': np.random.choice(cup_color_choices),
                'bin_color': np.random.choice(bin_color_choices)
            }
            
        # "put n ... cups (and m ... cups) in the ... dish"
        elif task_type == 'kitting':
            task = "put {first_cup_desc} {second_cup_desc} in the {bin_color} dish"

            # only one type of cups, so no second cup desc
            if len(set(cup_colors)) == 1:
                second_cup_desc = ''
                first_cup_desc = _genCupDesc(n_cups, cup_colors[0])
            
            else:
                colors = ['red','blue']
                np.random.shuffle(colors)
                first_cup_desc = _genCupDesc(cup_colors.count(colors[0]), colors[0])
                second_cup_desc = np.random.choice(['and ' + _genCupDesc(cup_colors.count(colors[1]), colors[1]), ''])
            
            bin_color_choices = []
            if n_bins == 1:
                bin_color_choices.append('')
            for bin_color in ['yellow', 'red', 'green']:
                if bin_colors.count(bin_color) > 0:
                    bin_color_choices.append(bin_color)
            descs = {
                'first_cup_desc': first_cup_desc,
                'second_cup_desc': second_cup_desc,
                'bin_color': np.random.choice(bin_color_choices)
            }
        
        # format task string
        task = task.format(**descs)
        task = ' '.join(task.split())
        return task

def _genCupDesc(count, cup_color):
    cup_desc = "{cup_count} {cup_color} {cup_or_cups}"
    cup_count_choices = ['the', 'one']
    count = np.random.randint(1,count+1)
    if count == 1:
        cup_count = np.random.choice(['the','one'])
        cup_or_cups = 'cup'
    elif count == 2:
        cup_count = np.random.choice(['the', 'one', 'two'])
        cup_or_cups = 'cup' if cup_count == 'one' else 'cups' if cup_count == 'two' else np.random.choice(['cup','cups'])
    cup_desc = cup_desc.format(**{'cup_count':cup_count, 'cup_color':cup_color, 'cup_or_cups':cup_or_cups})
    return cup_desc
    

  
if __name__ == "__main__":
    sim = Simulator()
    if COLLECT_DATA:
        print(f"Collecting {NUM_SAMPLES} Samples for Sorting..")
        sim.collectData(num_samples=NUM_SAMPLES, eval_type='sorting')
        print('#################################################')
        print(f"Collecting {NUM_SAMPLES} Samples for Kitting..")
        sim.collectData(num_samples=NUM_SAMPLES, eval_type='kitting')
        print("DATA COLLECTION COMPLETE")
    elif RUN_ON_TEST_DATA:
        sim.evalDirect(runs=NUM_TESTED_DATA)
    else:
        sim.runManually()
    sim.shutdown()
    