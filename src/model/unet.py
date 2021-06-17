import tensorflow as tf

from tensorpack import *
from tensorpack.models import BatchNorm, BNReLU, Conv2D, MaxPooling, FixedUnPooling
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary

from .utils import *

import sys
sys.path.append("..") # adds higher directory to python modules path.
try: # HACK: import beyond current level, may need to restructure
    from config import Config
except ImportError:
    assert False, 'Fail to import config.py'
    
    
def res_blk(name, l, ch, ksize, count, split=1, strides=1, freeze=False):
    ch_in = l.get_shape().as_list()
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block' + str(i)):  
                x = l if i == 0 else BNReLU('preact', l)
                x = Conv2D('conv1', x, ch[0], ksize[0], activation=BNReLU)
                x = Conv2D('conv2', x, ch[1], ksize[1], split=split, 
                                strides=strides if i == 0 else 1, activation=BNReLU)
                x = Conv2D('conv3', x, ch[2], ksize[2], activation=tf.identity)
                if (strides != 1 or ch_in[1] != ch[2]) and i == 0:
                    l = Conv2D('convshortcut', l, ch[2], 1, strides=strides)
                x = tf.stop_gradient(x) if freeze else x
                l = l + x
        # end of each group need an extra activation
        l = BNReLU('bnlast',l)  
    return l

def encoder(i, freeze):
    """
    Pre-activated ResNet50 Encoder
    """
    d1 = Conv2D('conv0',  i, 64, 7, padding='valid', strides=1, activation=BNReLU)
    d1 = res_blk('group0', d1, [ 64,  64,  256], [1, 3, 1], 3, strides=1, freeze=freeze)                       

    d2 = res_blk('group1', d1, [128, 128,  512], [1, 3, 1], 4, strides=2, freeze=freeze)
    d2 = tf.stop_gradient(d2) if freeze else d2

    d3 = res_blk('group2', d2, [256, 256, 1024], [1, 3, 1], 6, strides=2, freeze=freeze)
    d3 = tf.stop_gradient(d3) if freeze else d3

    d4 = res_blk('group3', d3, [512, 512, 2048], [1, 3, 1], 3, strides=2, freeze=freeze)
    d4 = tf.stop_gradient(d4) if freeze else d4

    d4 = Conv2D('conv_bot',  d4, 1024, 1, padding='same')
    return d1, d2, d3, d4
    
class Graph(ModelDesc, Config):
    def __init__(self):
        super(Graph, self).__init__()
        assert tf.test.is_gpu_available()
        self.data_format = 'channels_first'

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None] + self.train_input_shape + [3], 'images'),
                InputDesc(tf.float32, [None] + self.train_mask_shape  + [None], 'truemap-coded')]
          
    # for node to receive manual info such as learning rate.
    def add_manual_variable(self, name, init_value, summary=True):
        var = tf.get_variable(name, initializer=init_value, trainable=False)
        if summary:
            tf.summary.scalar(name + '-summary', var)
        return

    def _get_optimizer(self):
        with tf.variable_scope("", reuse=True):
            lr = tf.get_variable('learning_rate')
        opt = self.optimizer(learning_rate=lr)
        return opt

    def _build_graph(self, inputs):
        
        is_training = get_current_tower_context().is_training

        images, truemap_coded = inputs
        orig_imgs = images

        pen_map = truemap_coded[...,-1]
        if hasattr(self, 'type_classification') and self.type_classification:
            true = truemap_coded[...,1]
        else:
            true = truemap_coded[...,0]            
        true = tf.cast(true, tf.int32)
        true = tf.identity(true, name='truemap')
        one  = tf.one_hot(true, self.nr_types if self.type_classification else self.nr_classes, axis=-1)
        true = tf.expand_dims(true, axis=-1)

        def encoder_blk(name, feat_in, num_feats, has_down=False):
            with tf.variable_scope(name):
                feat = feat_in if not has_down else MaxPooling('pool1', feat_in, 2, strides=2, padding= 'same') 
                feat = Conv2D('conv_1', feat, num_feats, 3, padding='valid', strides=1, activation=tf.nn.relu)
                feat = Conv2D('conv_2', feat, num_feats, 3, padding='valid', strides=1, activation=tf.nn.relu)
                return feat                

        def decoder_blk(name, feat_in, num_feats, shorcut):
            with tf.variable_scope(name):
                in_ch = feat_in.get_shape().as_list()[1]
                feat = Conv2DTranspose('us', feat_in, in_ch, 2, strides=(2, 2), padding='same', activation=tf.identity)
                feat = tf.concat([feat, shorcut], axis=1)
                feat = Conv2D('conv_1', feat, num_feats, 3, padding='valid', strides=1, activation=tf.nn.relu)
                feat = Conv2D('conv_2', feat, num_feats, 3, padding='valid', strides=1, activation=tf.nn.relu)
                return feat

        #### Xavier initializer
        with argscope(Conv2D, activation=tf.identity, use_bias=False,
                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()), \
                argscope([Conv2D, BatchNorm, MaxPooling, Conv2DTranspose], data_format=self.data_format):

            i = tf.transpose(images, [0, 3, 1, 2])
            i = i if not self.input_norm else i / 255.0

#             d1 = encoder_blk('d1',  i,   64, has_down=False)
#             d2 = encoder_blk('d2', d1,  128, has_down=True)
#             d3 = encoder_blk('d3', d2,  256, has_down=True)
#             d4 = encoder_blk('d4', d3,  512, has_down=True)
#             d4 = tf.layers.dropout(d4, rate=0.5, seed=5, training=is_training)
#             d5 = encoder_blk('d5', d4, 1024, has_down=True)
#             d5 = tf.layers.dropout(d5, rate=0.5, seed=5, training=is_training)

            d1, d2, d3, d4 = encoder(i, freeze=False)

            d1 = crop_op(d1, (24,24))
            d2 = crop_op(d2,  (8,8) )
#             d3 = crop_op(d3,  (4,4) )
#             d4 = crop_op(d4,   (8,8)  )

            feat = decoder_blk('u3',   d4, 512, d3)
            feat = decoder_blk('u2', feat, 256, d2)
            feat = decoder_blk('u1', feat, 128, d1)
#             feat = decoder_blk('u1', feat,  64, d1)

            logi = Conv2D('conv_out', feat, 
                                self.nr_types if self.type_classification else self.nr_classes, 
                                1, use_bias=True, activation=tf.identity)
            logi = tf.transpose(logi, [0, 2, 3, 1])
            soft = tf.nn.softmax(logi, axis=-1)

            if self.type_classification:
                prob_np = tf.reduce_sum(soft[...,1:], axis=-1, keepdims=True)
                prob_np = tf.identity(prob_np, name='predmap-prob-np')
                predmap_coded = tf.concat([soft, prob_np], axis=-1)
            else:
                prob_np = tf.identity(soft[...,1], name='predmap-prob')
                prob_np = tf.expand_dims(prob_np, axis=-1)
                predmap_coded = prob_np

            # * channel ordering: type-map, segmentation map
            # encoded so that inference can extract all output at once
            predmap_coded = tf.identity(predmap_coded, name='predmap-coded')

        ####
        if is_training:
            ######## LOSS
            ### classification loss
            loss_bce = categorical_crossentropy(soft, one)
            loss_bce = tf.reduce_mean(loss_bce * pen_map, name='loss-bce')
            add_moving_summary(loss_bce)

            wd_loss = regularize_cost('.*/W', l2_regularizer(1.0e-5), name='l2_wd_loss')
            add_moving_summary(wd_loss)
            self.cost = loss_bce + wd_loss

            add_param_summary(('.*/W', ['histogram']))   # monitor W

            #### logging visual sthg
            orig_imgs = tf.cast(orig_imgs  , tf.uint8)
            orig_imgs = crop_op(orig_imgs, (34, 34), "channels_last")
            tf.summary.image('input', orig_imgs, max_outputs=1)
    
            pred = colorize(prob_np[...,0], cmap='jet')
            true = colorize(true[...,0], cmap='jet')
            pen_map = colorize(pen_map, cmap='jet')

            viz = tf.concat([orig_imgs, pred, true, pen_map], 2)

            tf.summary.image('output', viz, max_outputs=1)

        return
