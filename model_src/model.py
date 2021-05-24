# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import tensorflow as tf
import pathlib
from model_src.attention import TopDownAttention
from model_src.glove import GloveEmbeddings
from model_src.dmp import DynamicMovementPrimitive
from model_src.basismodel import BasisModel
from model_src.feedbackcontroller import FeedbackController
from dain_object_detector import show_bounding_boxes
import numpy as np
from semantic_parser import semantic_parser
import re
import time


class PolicyTranslationModel(tf.keras.Model):
    def __init__(self, od_path, glove_path, special=None):
        super(PolicyTranslationModel, self).__init__(name="policy_translation")
        self.units               = 32
        self.output_dims         = 7
        self.basis_functions     = 11
        self.dictionary    = self._loadDictionary(glove_path)
        self.regex         = re.compile('[^a-z ]')

        if od_path != "":
            od_path    = pathlib.Path(od_path)/"saved_model" 
            self.frcnn = tf.saved_model.load(str(od_path))
            self.frcnn = self.frcnn.signatures['serving_default']
            self.frcnn.trainable = False

        self.embedding = GloveEmbeddings(file_path=glove_path)
        self.lng_gru   = tf.keras.layers.GRU(units=self.units)

        self.attention = TopDownAttention(units=64)

        self.dout      = tf.keras.layers.Dropout(rate=0.25)

        # Units needs to be divisible by 7
        self.pt_global = tf.keras.layers.Dense(units=42, activation=tf.keras.activations.relu)

        self.pt_dt_1   = tf.keras.layers.Dense(units=self.units * 2, activation=tf.keras.activations.relu)
        self.pt_dt_2   = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.hard_sigmoid)

        self.controller = tf.keras.layers.RNN(
            FeedbackController(
                robot_state_size = self.units, 
                rnn_state_size   = (tf.TensorShape([self.output_dims]), tf.TensorShape([self.units])),
                dimensions       = self.output_dims, 
                basis_functions  = self.basis_functions,
                special          = None
            ), 
        return_sequences=True)

        self.features_for_bounding_box = None
        self.subtasks = []
        self.subtask_idx = 0
        self.cur_subtask = None
        self.subtask_attn = None
        self.subtask_embedding = None
        self.phase = 0.0
    
    def reset_state(self):
        self.subtasks = []
        self.subtask_idx = 0
        self.cur_subtask = None
        self.subtask_attn = None
        self.subtask_embedding = None
        self.phase = 0.0
    
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
    
    # @tf.function
    def generate_subtask_embedding(self, instruction, features, robot):
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
    
    # @tf.function
    def prep_controller_call(self, robot, batch_size, use_dropout):
        pt          = self.pt_global(self.subtask_embedding)
        pt          = self.dout(pt, training=tf.convert_to_tensor(use_dropout))
        dmp_dt      = self.pt_dt_2(self.pt_dt_1(pt)) + 0.1 # 0.1 prevents division by 0, just in case
        # dmp_dt      = d_out[2]

        # Run the low-level controller
        start_joints  = robot[:,0,:]
        initial_state = [
            start_joints,
            tf.zeros(shape=[batch_size, self.units], dtype=tf.float32)
        ]
        return dmp_dt, initial_state
    
    # @tf.function
    def call(self, inputs, training=False, use_dropout=True):
        print('---call---')
        ss = time.time()
        if training:
            use_dropout = True
        
        if self.features_for_bounding_box:
            show_bounding_boxes(self.features_for_bounding_box[0],
                                self.features_for_bounding_box[1],
                                self.features_for_bounding_box[2],
                                self.features_for_bounding_box[3])

        language   = inputs[0]
        features   = inputs[1]
        robot      = inputs[2]
        # dmp_state  = inputs[3]
        tf.config.experimental_run_functions_eagerly(True)

        if self.subtasks == []: #S_0
            s = time.time()
            subtasks = semantic_parser(language)
            print('generated subtasks for model: ', round(time.time()-s, 3), 'seconds')
            print(subtasks)
            self.subtasks = subtasks
            # catch case where command is malformed / no subtasks are generated from command
            if self.subtasks == []:
                return self.old_call(inputs, training=training, use_dropout=use_dropout)

        # Call word embedding only once at the beginning for efficiency
        if self.cur_subtask is None:
            s = time.time()
            cur_subtask = self.subtasks[self.subtask_idx]
            # print('current subtask:',cur_subtask)
            # From service.py: convert to GloVe word embeddings
            cur_subtask = self.tokenize(cur_subtask)
            cur_subtask = cur_subtask + [0] * (15-len(cur_subtask))
            self.cur_subtask = tf.convert_to_tensor(np.tile([cur_subtask],[250, 1]), dtype=tf.int64)
            #
            print('embedded current subtask instruction: ',round(time.time()-s, 3),'seconds')
        
        # input_data = (
        #     self.cur_subtask,
        #     features,
        #     robot
        # )
        # return self.old_call(input_data, training=training, use_dropout=use_dropout)

        batch_size = tf.shape(self.cur_subtask)[0]

        if self.subtask_embedding is None:
            s = time.time()
            instruction  = self.embedding(self.cur_subtask)
            instruction  = self.lng_gru(inputs=instruction, training=training) 

            # Calculate attention for current subtask
            a = self.attention((instruction, features))
            print('embedding and calculating attention for subtask:',round(time.time()-s, 3),'seconds')
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
            s = time.time()
            self.subtask_attn = tf.numpy_function(random_choose, [a], tf.float32)
            self.subtask_attn = tf.convert_to_tensor(self.subtask_attn, dtype=tf.float32)
            print('generate random choose mask for subtask:',round(time.time()-s, 3),'seconds')

            s = time.time()
            # atn_w = tf.expand_dims(self.subtask_attn, 2)
            # atn_w = tf.tile(atn_w, [1, 1, 5])
            # # Compress image features and apply attention
            # cfeatures = tf.math.multiply(atn_w, features)
            # cfeatures = tf.math.reduce_sum(cfeatures, axis=1)
            # # Add the language to the mix again. Possibly usefull to predict dt
            # start_joints  = robot[:,0,:]
            # cfeatures = tf.keras.backend.concatenate((cfeatures, instruction, start_joints), axis=1)
            # # Save subtask embedding
            # self.subtask_embedding = cfeatures
            self.generate_subtask_embedding(instruction, features, robot)
            print('generate subtask embedding:',round(time.time()-s, 3),'seconds')

        # Policy Translation: Create weight + goal for DMP
        s = time.time()
        # pt          = self.pt_global(self.subtask_embedding)
        # pt          = self.dout(pt, training=tf.convert_to_tensor(use_dropout))
        # dmp_dt      = self.pt_dt_2(self.pt_dt_1(pt)) + 0.1 # 0.1 prevents division by 0, just in case
        # # dmp_dt      = d_out[2]

        # # Run the low-level controller
        # start_joints  = robot[:,0,:]
        # initial_state = [
        #     start_joints,
        #     tf.zeros(shape=[batch_size, self.units], dtype=tf.float32)
        # ]
        dmp_dt, initial_state = self.prep_controller_call(robot, batch_size, use_dropout)
        print('prep for calling controller:',round(time.time()-s, 3),'seconds')
        s = time.time()
        generated, subtask_phase, weights = self.controller(inputs=robot, constants=(self.subtask_embedding, dmp_dt), initial_state=initial_state, training=training)
        # subtask_phase = tf.compat.v1.Print(subtask_phase, [subtask_phase], "subtask_phase: ", summarize=6*5)
        print('controller model:',round(time.time()-s, 3),'seconds')
        # s = time.time()
        # subtask_phase     = tf.math.reduce_mean(subtask_phase, axis=0).numpy()
        # subtask_phase     = subtask_phase[-1,0]
        # self.phase = subtask_phase / (len(self.subtasks)+0.1) # prevent zero division error
        # # subtask_phase = 0.05
        # # self.phase += subtask_phase # TODO overriden for tf.function test
        # print('calculate subtask phase:',round(time.time()-s, 3),'seconds')
        # # check state condition
        # if subtask_phase > 0.95:
        #     # move onto the next subtask
        #     print('moving onto next subtask..')
        #     self.subtask_idx += 1
        #     self.cur_subtask = None
        #     self.subtask_attn = None
        #     self.subtask_embedding = None
        
        tf.config.experimental_run_functions_eagerly(False)
        
        # if self.phase > 0.95 or self.subtask_idx >= len(self.subtasks): #S_f
        #     print('-----DONE WITH ALL SUBTASKS-----')
        #     self.reset_state()
        #     print('MODEL TOOK:',round(time.time()-ss, 3),'seconds')
        #     return generated, (self.subtask_attn, dmp_dt, 1.0, weights)
        # print('MODEL TOOK:',round(time.time()-ss, 3),'seconds')
        # return generated, (self.subtask_attn, dmp_dt, self.phase, weights)
        return generated, (self.subtask_attn, dmp_dt, subtask_phase, weights)

           
    @tf.function
    def old_call(self, inputs, training=False, use_dropout=True):
        if training:
            use_dropout = True

        language   = inputs[0]
        features   = inputs[1]
        # local      = features[:,:,:5]
        robot      = inputs[2]
        # dmp_state  = inputs[3]
        batch_size = tf.shape(language)[0]

        language  = self.embedding(language)
        language  = self.lng_gru(inputs=language, training=training) 

        # Calculate attention and expand it to match the feature size
        a = self.attention((language, features))
        # atn = tf.nn.sigmoid(a)
        # atn = a
        # TODO: selection process [0,1,0,0,0,0]
        def random_choose(a, thresh=0.9):
            sig = tf.nn.sigmoid(a)
            # randomly chooses the column index where sigmoid value >= thresh
            try:
                idx = np.random.choice(np.where(sig[0]>=thresh)[0])
            except:
                idx = 0
            # idx = 1
            # print('Mask options:',sig[0])
            z = np.zeros((a.shape[1]), dtype="float32")
            z[idx] = 1
            mask = np.tile(z, (a.shape[0],1))
            return mask
            # return tf.math.multiply(a, mask)
        
        # if not self.mask:
        #     self.mask = tf.numpy_function(random_choose, [a], tf.float32)
        # atn = tf.math.multiply(tf.nn.sigmoid(a), mask)
        # atn = tf.numpy_function(random_choose, [a], tf.float32)  ## Uncomment this for selection process
        # atn = tf.convert_to_tensor(atn, dtype=tf.float32)  # Uncomment this for selection process
        # atn = tf.compat.v1.Print(atn, [atn], "atn after convert: ", summarize=6*5)
        atn = a  ## This is the original attn output

        atn_w = tf.expand_dims(atn, 2)
        atn_w = tf.tile(atn_w, [1, 1, 5])
        # Compress image features and apply attention
        cfeatures = tf.math.multiply(atn_w, features)
        # print(atn_w)
        # atn_w = tf.compat.v1.Print(atn_w, [atn_w], "atn_w: ", summarize=6*5*3)
        # TODO: list of commands here
        # TODO: low-level call function
        # how is the confidence computed? Does it add to 1?
        cfeatures = tf.math.reduce_sum(cfeatures, axis=1)

        # Add the language to the mix again. Possibly usefull to predict dt
        start_joints  = robot[:,0,:]
        cfeatures = tf.keras.backend.concatenate((cfeatures, language, start_joints), axis=1)
        # cfeatures = tf.compat.v1.Print(cfeatures, [cfeatures], "cfeatures: ", summarize=44)

        # Policy Translation: Create weight + goal for DMP
        pt          = self.pt_global(cfeatures)
        pt          = self.dout(pt, training=tf.convert_to_tensor(use_dropout))
        dmp_dt      = self.pt_dt_2(self.pt_dt_1(pt)) + 0.1 # 0.1 prevents division by 0, just in case
        # dmp_dt      = d_out[2]

        # Run the low-level controller
        initial_state = [
            start_joints,
            tf.zeros(shape=[batch_size, self.units], dtype=tf.float32)
        ]
        generated, phase, weights = self.controller(inputs=robot, constants=(cfeatures, dmp_dt), initial_state=initial_state, training=training)
        # TODO check condition
        # phase /= len(self.subtasks)
        return generated, (atn, dmp_dt, phase, weights)
    
    def getVariables(self, step=None):
        return self.trainable_variables
    
    def getVariablesFT(self):
        variables = []
        variables += self.pt_w_1.trainable_variables
        variables += self.pt_w_2.trainable_variables
        variables += self.pt_w_3.trainable_variables
        return variables
    
    def saveModelToFile(self, add):
        self.save_weights("Data/Model/" + add + "policy_translation")

    def saveBoundingBoxInfo(self, image, features):
        print('received bounding box info')
        scores   = features["detection_scores"][0, :6].numpy().astype(dtype=np.float32)
        classes  = features["detection_classes"][0, :6].numpy().astype(dtype=np.int32)
        boxes    = features["detection_boxes"][0, :6, :].numpy().astype(dtype=np.float32)
        self.features_for_bounding_box = [image, boxes, classes, scores]