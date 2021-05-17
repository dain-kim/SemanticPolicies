# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import tensorflow as tf

class TopDownAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwarg):
        super(TopDownAttention, self).__init__(name="attention", **kwarg)
        self.units   = units  # 64

    def build(self, input_shape):
        self.w1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.units, activation=tf.keras.activations.tanh))
        self.w2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=self.units, activation=tf.keras.activations.sigmoid))
        self.wt = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear, use_bias=False))

    def call(self, inputs, training=None):
        language = inputs[0]
        # language = tf.compat.v1.Print(language, [language], "language: ", summarize=32*2)
        features = inputs[1]
        # features = tf.compat.v1.Print(features, [features], "features: ", summarize=6*5)
        k        = tf.shape(features)[1]
        # print('language input', language.get_shape())
        # print('features input', features.get_shape())

        language = tf.tile(tf.expand_dims(language, 1), [1, k, 1])            # bxkxm
        att_in   = tf.keras.backend.concatenate((language, features), axis=2) # bxkx(m+n)
        # print('language',language.get_shape())
        # print('att_in',att_in.get_shape())
        
        y_1 = self.w1(att_in)
        # print('y_1', y_1.get_shape())
        y_2 = self.w2(att_in)
        # print('y_2', y_2.get_shape())
        y   = tf.math.multiply(y_1, y_2)
        # print('y', y.get_shape())
        a   = self.wt(y)
        # print('a (after applying linear self.wt)', a.get_shape())
        a   = tf.squeeze(a, axis=2)
        # print('a (after squeezing)', a.get_shape())
        return a  ## Uncomment this for model.py selection process
        soft = tf.nn.softmax(a)
        # print('softmax', soft.get_shape())
        return soft


        # TODO: check the output a before the softmax
        a_temp = a[0]
        a_temp = tf.nn.sigmoid(a_temp)
        a_temp = tf.compat.v1.Print(a_temp, [a_temp], "a_temp: ", summarize=6*5)
        # print(tf.size(a_temp))
        # print(tf.rank(a_temp))
        # print(tf.size(a))
        # print(tf.rank(a))
        # print(tf.shape(a))
        # print(a.get_shape())
        # print(tf.executing_eagerly())
        # m = tf.reduce_any(tf.greater(tf.expand_dims(a_temp, 1), 0), axis=1)
        # # print(tf.math.reduce_sum(tf.cast(m, tf.int32)).numpy())
        # a_reduced = tf.boolean_mask(a[0], m)
        # a_temp = tf.compat.v1.Print(a_temp, [a_temp], "a: ", summarize=6*5)
        # aaa = a_reduced.eval(session=tf.compat.v1.Session())
        # print(aaa)
        # print(a_reduced)
        # tmp = tf.constant([0.5,0.4,-0.9])
        # print(tmp.numpy())
        # print(a_reduced.numpy())
        
        # def shape(tensor):
        #     if tensor:
        #         s = tensor.get_shape()
        #         return tuple([s[i].value for i in range(0, len(s))])
        # print(a_reduced.get_shape())
        # print(shape(a_temp))
        
        # a_reduced = tf.compat.v1.Print(a_reduced, [a_reduced], "a_reduced: ", summarize=6)
        # soft = tf.nn.softmax(a)
        # TODO: heatmap function here using the output of softmax
        # soft = tf.compat.v1.Print(soft, [soft], "SOFTMAX: ", summarize=5*6)
        # return soft

    def get_config(self):
        config = super(TopDownAttention, self).get_config()
        config.update({'units': self.units})
        return config