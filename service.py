#!/usr/bin/env python

# original @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import rclpy
from policy_translation.srv import NetworkPT, TuneNetwork
from model_src.model import PolicyTranslationModel
from utils.network import Network
from utils.tf_util import trainOnCPU, limitGPUMemory
from utils.intprim.gaussian_model import GaussianModel
import tensorflow as tf
import numpy as np
import re
from cv_bridge import CvBridge, CvBridgeError
import cv2
import matplotlib.pyplot as plt
from utils.intprim.gaussian_model import GaussianModel
import glob
import json
import pickle
import copy
import time

from dain_object_detector import show_bounding_boxes
from semantic_parser import semantic_parser


# Force TensorFlow to use the CPU
FORCE_CPU    = True
# Use dropout at run-time for stochastif-forward passes
USE_DROPOUT  = True
# Where can we find the trained model?
MODEL_PATH   = "../GDrive/model/policy_translation"
# Where is a pre-trained faster-rcnn?
FRCNN_PATH   = "../GDrive/rcnn"
# Where are the GloVe word embeddings?
GLOVE_PATH   = "../GDrive/glove.6B.50d.txt"
# Where is the normalization of the dataset?
NORM_PATH    = "../GDrive/normalization_v2.pkl"

if FORCE_CPU:
    trainOnCPU()
else:
    limitGPUMemory()

print("Running Policy Translation Model")
model = PolicyTranslationModel(
    od_path=FRCNN_PATH,
    glove_path=GLOVE_PATH,
    special=None 
)

bs = 2
model((
    np.ones((bs, 15), dtype=np.int64),
    np.ones((bs, 6, 5), dtype=np.float32),
    np.ones((bs, 500, 7), dtype=np.float32)
))
model.load_weights(MODEL_PATH)
model.summary()

class NetworkService():
    def __init__(self):
        self.dictionary    = self._loadDictionary(GLOVE_PATH)
        self.regex         = re.compile('[^a-z ]')
        self.bridge        = CvBridge()
        self.history       = []
        rclpy.init(args=None)
        self.node = rclpy.create_node("neural_network")
        self.service_nn = self.node.create_service(NetworkPT,   "/network",      self.cbk_network_dmp_ros2)
        self.normalization = pickle.load(open(NORM_PATH, mode="rb"), encoding="latin1")
        print("Ready")
        self.reset_state()

    def runNode(self):
        while rclpy.ok():
            rclpy.spin_once(self.node)
        self.node.destroy_service(self.service_nn)
        self.node.destroy_service(self.service_tn)
        rclpy.shutdown()

    def _loadDictionary(self, file):
        __dictionary = {}
        __dictionary[""] = 0 # Empty string
        fh = open(file, "r", encoding="utf-8")
        for line in fh:
            if len(__dictionary) >= 300000:
                break
            tokens = line.strip().split(" ")
            __dictionary[tokens[0]] = len(__dictionary)
        fh.close()
        return __dictionary

    def tokenize(self, language):
        voice  = self.regex.sub("", language.strip().lower())
        tokens = []
        for w in voice.split(" "):
            idx = 0
            try:
                idx = self.dictionary[w]
            except:
                print("Unknown word: " + w)
            tokens.append(idx)
        return tokens
    
    def normalize(self, value, v_min, v_max):
        if (value.shape[1] != v_min.shape[0] or v_min.shape[0] != v_max.shape[0] or 
            len(value.shape) != 2 or len(v_min.shape) != 1 or len(v_max.shape) != 1):
            raise ArrayDimensionMismatch()
        value = np.copy(value)
        v_min = np.tile(np.expand_dims(v_min, 0), [value.shape[0], 1])
        v_max = np.tile(np.expand_dims(v_max, 0), [value.shape[0], 1])
        value = (value - v_min) / (v_max - v_min)
        return value

    def interpolateTrajectory(self, trj, target):
        current_length = trj.shape[0]
        dimensions     = trj.shape[1]
        result         = np.zeros((target, trj.shape[1]), dtype=np.float32)
              
        for i in range(dimensions):
            result[:,i] = np.interp(np.linspace(0.0, 1.0, num=target), np.linspace(0.0, 1.0, num=current_length), trj[:,i])
        
        return result

    def cbk_network_dmp_ros2(self, req, res):
        res.trajectory, res.confidence, res.timesteps, res.weights, res.phase, res.features, res.attn = self.cbk_network_dmp(req)
        return res
    
    def imgmsg_to_cv2(self, img_msg, desired_encoding="passthrough"):   
        if img_msg.encoding != "8UC3":     
            self.node.get_logger().info("Unrecognized image type: " + encoding)
            exit(0)
        dtype      = "uint8"
        n_channels = 3

        dtype = np.dtype(dtype)
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')

        img_buf = np.asarray(img_msg.data, dtype=dtype) if isinstance(img_msg.data, list) else img_msg.data

        if n_channels == 1:
            im = np.ndarray(shape=(img_msg.height, img_msg.width),
                            dtype=dtype, buffer=img_buf)
        else:
            im = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                            dtype=dtype, buffer=img_buf)
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            im = im.byteswap().newbyteorder()

        if desired_encoding == 'passthrough':
            return im

        from cv_bridge.boost.cv_bridge_boost import cvtColor2

        try:
            res = cvtColor2(im, img_msg.encoding, desired_encoding)
        except RuntimeError as e:
            raise CvBridgeError(e)

        return res

    def reset_state(self):
        self.history = []
        self.sfp_history = []
        self.req_step = 1
        self.task_embedding = None

    def cbk_network_dmp(self, req):
        if req.reset: # cnt = 1
            self.reset_state()
            self.raw_a = np.array([])
            try:
                image = self.imgmsg_to_cv2(req.image)
            except CvBridgeError as e:
                print(e)

            # Image processing
            image_features = model.frcnn(tf.convert_to_tensor([image], dtype=tf.uint8))
            scores   = image_features["detection_scores"][0, :6].numpy().astype(dtype=np.float32)
            scores   = [0.0 if v < 0.5 else 1.0 for v in scores.tolist()]
            classes  = image_features["detection_classes"][0, :6].numpy().astype(dtype=np.int32)
            classes  = [v * scores[k] for k, v in enumerate(classes.tolist())]
            boxes    = image_features["detection_boxes"][0, :6, :].numpy().astype(dtype=np.float32)
            
            # show_bounding_boxes(image, boxes, classes, scores, save=True)
            
            self.features = np.concatenate((np.expand_dims(classes,1), boxes), axis=1)

            # Tokenize language input
            language = self.tokenize(req.language)
            self.language = language + [0] * (15-len(language))

            self.language_tf = tf.convert_to_tensor(np.tile([self.language],[250, 1]), dtype=tf.int64)
            self.features_tf = tf.convert_to_tensor(np.tile([self.features],[250, 1, 1]), dtype=tf.float32)

            # Attention network
            sentence_embedding, atn = model.get_attention(self.language_tf, self.features_tf, training=tf.constant(False))
            self.raw_a = atn[0].numpy()
            
            # Task object selector
            atn = tf.numpy_function(random_choose, [atn], tf.float32)
            atn = tf.convert_to_tensor(atn, dtype=tf.float32)

            # Create task embedding (language + features)
            atn_w = tf.expand_dims(atn, 2)
            atn_w = tf.tile(atn_w, [1, 1, 5])
            # Compress image features and apply attention
            task_embedding = tf.math.multiply(atn_w, self.features_tf)
            task_embedding = tf.math.reduce_sum(task_embedding, axis=1)
            # Add the language to the mix again. Possibly usefull to predict dt
            _history = [list(req.robot)]
            _robot = np.asarray(_history, dtype=np.float32)
            _tmp = tf.convert_to_tensor(np.tile([_robot],[250, 1, 1]), dtype=tf.float32)
            start_joints  = _tmp[:,0,:]
            self.task_embedding = tf.keras.backend.concatenate((task_embedding, sentence_embedding, start_joints), axis=1)


        self.history.append(list(req.robot))
        robot           = np.asarray(self.history, dtype=np.float32)
        self.robot_tf = tf.convert_to_tensor(np.tile([robot],[250, 1, 1]), dtype=tf.float32)

        if self.task_embedding is None:
            return ([], [], 0, [], 0.0, self.features.flatten().tolist(), self.raw_a.tolist())
        
        # call the model with the current embedding
        input_data = (
            self.language_tf,
            self.features_tf,
            self.robot_tf
        )
        generated, (atn, dmp_dt, phase, weights) = model.new_call(input_data, self.task_embedding, training=tf.constant(False), use_dropout=tf.constant(True))

        self.trj_gen    = tf.math.reduce_mean(generated, axis=0).numpy()
        self.trj_std    = tf.math.reduce_std(generated, axis=0).numpy()
        self.timesteps  = int(tf.math.reduce_mean(dmp_dt).numpy() * 500)
        self.b_weights  = tf.math.reduce_mean(weights, axis=0).numpy()
        subtask_phase     = tf.math.reduce_mean(phase, axis=0).numpy()
        subtask_phase     = subtask_phase[-1,0]
        
        if 'pour' in req.language and self.trj_gen[-1][5] > 0.55:
            subtask_phase = 1
            self.req_step = 101

        # Determine if task is complete
        if subtask_phase > 0.98 and self.req_step > 100:
            trj_len    = len(self.sfp_history)
            basismodel = GaussianModel(degree=11, scale=0.012, observed_dof_names=("Base","Shoulder","Ellbow","Wrist1","Wrist2","Wrist3","Gripper"))
            domain     = np.linspace(0, 1, trj_len, dtype=np.float64)
            trajectories = []
            for i in range(trj_len):
                trajectories.append(np.asarray(basismodel.apply_coefficients(domain, self.sfp_history[i].flatten())))
            trajectories = np.asarray(trajectories)
            np.save("trajectories", trajectories)
            np.save("history", self.history)

            gen_trajectory = []
            var_trj        = np.zeros((trj_len, trj_len, 7), dtype=np.float32)
            for w in range(trj_len):
                gen_trajectory.append(trajectories[w,w,:])
            gen_trajectory = np.asarray(gen_trajectory)
            np.save("gen_trajectory", gen_trajectory)            

            self.reset_state()
        
        self.req_step += 1
        return (self.trj_gen.flatten().tolist(), self.trj_std.flatten().tolist(), self.timesteps, self.b_weights.flatten().tolist(), float(subtask_phase), self.features.flatten().tolist(), self.raw_a.tolist())
    
    def idToText(self, id):
        names = ["", "Yellow Small Round", "Red Small Round", "Green Small Round", "Blue Small Round", "Pink Small Round",
                     "Yellow Large Round", "Red Large Round", "Green Large Round", "Blue Large Round", "Pink Large Round",
                     "Yellow Small Square", "Red Small Square", "Green Small Square", "Blue Small Square", "Pink Small Square",
                     "Yellow Large Square", "Red Large Square", "Green Large Square", "Blue Large Square", "Pink Large Square",
                     "Cup Red", "Cup Green", "Cup Blue"]
        return names[id]

def random_choose(a, thresh=0.9):
    # sig = tf.nn.sigmoid(a)
    # # randomly chooses the column index where sigmoid value >= thresh
    # try:
    #     idx = np.random.choice(np.where(sig[0]>=thresh)[0])
    # except:
    #     idx = 0
    idx = np.argmax(a[0])
    # print('Mask options:',sig[0])
    z = np.zeros((a.shape[1]), dtype="float32")
    z[idx] = 1
    mask = np.tile(z, (a.shape[0],1))
    return mask
    
if __name__ == "__main__":
    ot = NetworkService()
    ot.runNode()