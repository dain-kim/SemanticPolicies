#!/usr/bin/env python

# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

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

from dain_object_detector import show_bounding_boxes
import time
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
        res.trajectory, res.confidence, res.timesteps, res.weights, res.phase = self.cbk_network_dmp(req)
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

    def cbk_network_dmp(self, req):
        if req.reset:
            self.req_step = 0
            self.sfp_history = []
            self.first_call = True
            try:
                image = self.imgmsg_to_cv2(req.image)
            except CvBridgeError as e:
                print(e)
            ## Original language processing
            # language = self.tokenize(req.language)
            # self.language = language + [0] * (15-len(language))

            image_features = model.frcnn(tf.convert_to_tensor([image], dtype=tf.uint8))
            # print('type',image_features)

            scores   = image_features["detection_scores"][0, :6].numpy().astype(dtype=np.float32)
            scores   = [0.0 if v < 0.5 else 1.0 for v in scores.tolist()]

            classes  = image_features["detection_classes"][0, :6].numpy().astype(dtype=np.int32)
            classes  = [v * scores[k] for k, v in enumerate(classes.tolist())]

            boxes    = image_features["detection_boxes"][0, :6, :].numpy().astype(dtype=np.float32)
            
            # detect_objects(model, image, tf.convert_to_tensor([image], dtype=tf.uint8), boxes, classes)
            # show_bounding_boxes(image, boxes, classes, scores)
            # model.saveBoundingBoxInfo(image, image_features)
            
            self.features = np.concatenate((np.expand_dims(classes,1), boxes), axis=1)

            self.history  = []        

        self.history.append(list(req.robot)) 

        robot           = np.asarray(self.history, dtype=np.float32)
        self.input_data = (
            # tf.convert_to_tensor(np.tile([self.language],[250, 1]), dtype=tf.int64), ## Original language input in tensor form
            req.language,
            tf.convert_to_tensor(np.tile([self.features],[250, 1, 1]), dtype=tf.float32),
            tf.convert_to_tensor(np.tile([robot],[250, 1, 1]), dtype=tf.float32)
        )

        if self.first_call:
            self.prep(self.input_data, training=tf.constant(False))
            self.first_call = False
        
        s = time.time()
        generated, (atn, dmp_dt, phase, weights) = model.new_call(self.input_data, self.cur_subtask, self.subtask_embedding, training=tf.constant(False), use_dropout=tf.constant(True))
        print('model took', round(time.time() - s,2), 'seconds to run')
        
        self.trj_gen    = tf.math.reduce_mean(generated, axis=0).numpy()
        self.trj_std    = tf.math.reduce_std(generated, axis=0).numpy()
        self.timesteps  = int(tf.math.reduce_mean(dmp_dt).numpy() * 500)
        self.b_weights  = tf.math.reduce_mean(weights, axis=0).numpy()

        phase_value     = tf.math.reduce_mean(phase, axis=0).numpy()
        phase_value     = phase_value[-1,0]
        # phase_value = phase

        self.sfp_history.append(self.b_weights[-1,:,:])
        if phase_value > 0.94:
            print('moving onto next subtask..')
            self.subtask_idx += 1
            self.cur_subtask = None
            self.subtask_attn = None
            self.subtask_embedding = None
            if self.subtask_idx >= len(self.subtasks):
                print('-----DONE WITH ALL SUBTASKS-----')
        # if phase_value > 0.95 and len(self.sfp_history) > 100:
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

                self.sfp_history = []
                # self.first_call = True
                # model.reset_state()
                self.reset_state()
            else:
                self.prep(self.input_data, training=tf.constant(False))

        
        self.req_step += 1
        return (self.trj_gen.flatten().tolist(), self.trj_std.flatten().tolist(), self.timesteps, self.b_weights.flatten().tolist(), float(phase_value)) 
    
    def idToText(self, id):
        names = ["", "Yellow Small Round", "Red Small Round", "Green Small Round", "Blue Small Round", "Pink Small Round",
                     "Yellow Large Round", "Red Large Round", "Green Large Round", "Blue Large Round", "Pink Large Round",
                     "Yellow Small Square", "Red Small Square", "Green Small Square", "Blue Small Square", "Pink Small Square",
                     "Yellow Large Square", "Red Large Square", "Green Large Square", "Blue Large Square", "Pink Large Square",
                     "Cup Red", "Cup Green", "Cup Blue"]
        return names[id]
    
    def plotTrajectory(self, trj, error, image):
        fig, ax = plt.subplots(3,3)
        fig.set_size_inches(9, 9)

        for sp in range(7):
            idx = sp // 3
            idy = sp  % 3
            ax[idx,idy].clear()
            ax[idx,idy].plot(range(trj.shape[0]), trj[:,sp], alpha=0.5, color='mediumslateblue')
            ax[idx,idy].errorbar(range(trj.shape[0]), trj[:,sp], xerr=None, yerr=error[:,sp], alpha=0.1, fmt='none', color='mediumslateblue')
            ax[idx,idy].set_ylim([-0.1, 1.1])

        ax[2,1].imshow(image)

    def plotImageRegions(self, image_np, image_dict, atn):
        # Visualization of the results of a detection.
        tgt_object   = np.argmax(atn)
        num_detected = len([v for v in image_dict["detection_scores"][0] if v > 0.5]) 
        num_detected = min(num_detected, len(atn))
        for i in range(num_detected):
            ymin, xmin, ymax, xmax = image_dict['detection_boxes'][0][i,:]
            pt1 = (int(xmin*image_np.shape[1]), int(ymin*image_np.shape[0]))
            pt2 = (int(xmax*image_np.shape[1]), int(ymax*image_np.shape[0]))
            image_np = cv2.rectangle(image_np, pt1, pt2, (156, 2, 2), 1)
            if i == tgt_object:
                image_np = cv2.rectangle(image_np, pt1, pt2, (30, 156, 2), 2)
                image_np = cv2.putText(image_np, "{:.1f}%".format(atn[i] * 100), (pt1[0]-10, pt1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 156, 2), 2, cv2.LINE_AA)
            
        fig = plt.figure()
        plt.imshow(image_np)
    
    def prep(self, inputs, training=False, use_dropout=True):
        print('---SERVICE PREP---')
        s = time.time()
        if training:
            use_dropout = True
        
        language   = inputs[0]
        features   = inputs[1]
        robot      = inputs[2]
        # dmp_state  = inputs[3]

        if self.subtasks == []: #S_0
            subtasks = semantic_parser(language)
            self.subtasks = subtasks
            # catch case where command is malformed / no subtasks are generated from command
            if self.subtasks == []:
                return
                # return self.old_call(inputs, training=training, use_dropout=use_dropout)

        # Call word embedding only once at the beginning for efficiency
        if self.cur_subtask is None:
            cur_subtask = self.subtasks[self.subtask_idx]
            cur_subtask = self.tokenize(cur_subtask)
            cur_subtask = cur_subtask + [0] * (15-len(cur_subtask))
            self.cur_subtask = tf.convert_to_tensor(np.tile([cur_subtask],[250, 1]), dtype=tf.int64)

        if self.subtask_embedding is None:
            instruction  = model.embedding(self.cur_subtask)
            instruction  = model.lng_gru(inputs=instruction, training=training) 

            # Calculate attention for current subtask
            a = model.attention((instruction, features))
            def random_choose(a, thresh=0.9):
                sig = tf.nn.sigmoid(a)
                # randomly chooses the column index where sigmoid value >= thresh
                try:
                    idx = np.random.choice(np.where(sig[0]>=thresh)[0])
                except:
                    idx = 0
                # print('Mask options:',sig[0])
                z = np.zeros((a.shape[1]), dtype="float32")
                z[idx] = 1
                mask = np.tile(z, (a.shape[0],1))
                return mask
            self.subtask_attn = tf.numpy_function(random_choose, [a], tf.float32)
            self.subtask_attn = tf.convert_to_tensor(self.subtask_attn, dtype=tf.float32)

            self.generate_subtask_embedding(instruction, features, robot)
            print('prep:',round(time.time()-s, 3),'seconds')
    
    def generate_subtask_embedding(self, instruction, features, robot):
        # print('subtask embedding running on graph mode? ', not tf.executing_eagerly())
        atn_w = tf.expand_dims(self.subtask_attn, 2)
        atn_w = tf.tile(atn_w, [1, 1, 5])
        # Compress image features and apply attention
        cfeatures = tf.math.multiply(atn_w, features)
        cfeatures = tf.math.reduce_sum(cfeatures, axis=1)
        # Add the language to the mix again. Possibly usefull to predict dt
        start_joints  = robot[:,0,:]
        cfeatures = tf.keras.backend.concatenate((cfeatures, instruction, start_joints), axis=1)
        # Save subtask embedding
        self.subtask_embedding = cfeatures
    
    def reset_state(self):
        self.first_call = True
        self.subtasks = []
        self.subtask_idx = 0
        self.cur_subtask = None
        self.subtask_attn = None
        self.subtask_embedding = None
        self.phase = 0.0
        # self.batch_size = None
    
if __name__ == "__main__":
    ot = NetworkService()
    ot.runNode()