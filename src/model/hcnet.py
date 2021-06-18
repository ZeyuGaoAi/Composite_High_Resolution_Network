import tensorflow as tf

from tensorpack import *
from tensorpack.models import BatchNorm, BNReLU, Conv2D, MaxPooling, FixedUnPooling, LayerNorm
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary

from .utils import *

import sys
sys.path.append("..") # adds higher directory to python modules path.
try: # HACK: import beyond current level, may need to restructure
    from config import Config
except ImportError:
    assert False, 'Fail to import config.py'
    
####
def upsample2x(name, x):
    """
    Nearest neighbor up-sampling
    """
    return FixedUnPooling(
                name, x, 2, unpool_mat=np.ones((2, 2), dtype='float32'),
                data_format='channels_first')

def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)
    
def context_block(name, x, channel):
    
    def spatial_pool(x, channel):
        
        batch, height, width = tf.shape(x)[0], tf.shape(x)[2], tf.shape(x)[3]
        input_x = x
        # [N, C, H * W]
        input_x = tf.reshape(input_x, [batch, channel, height * width])
        # [N, 1, C, H * W]
        input_x = tf.expand_dims(input_x, axis=1)
        # [N, 1, H, W]
        context_mask = Conv2D('spatial_pool_conv', x, 1, 1, strides=1)
        # [N, 1, H * W]
        context_mask = tf.reshape(context_mask, [batch, 1, height * width])
        # [N, 1, H * W]
        context_mask = tf.nn.softmax(context_mask, axis=-1)
        # [N, 1, H * W, 1]
        context_mask = tf.expand_dims(context_mask, axis=-1)
        # [N, 1, C, 1]
        context = tf.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = tf.reshape(context, [batch, channel, 1, 1])
        
        return context
    
    def channel_add_conv(x, channel, ratio=8):
        
        batch, height, width = tf.shape(x)[0], tf.shape(x)[2], tf.shape(x)[3]
        out_x = x
        
        out_x = Conv2D('channel_add_conv_1', out_x, int(channel/ratio), 1, strides=1)
        out_x = LayerNorm('channel_add_layernorm', out_x)
        out_x = tf.nn.relu(out_x)
        out_x = Conv2D('channel_add_conv_2', out_x, channel, 1, strides=1)
        
        return out_x
    
    with tf.variable_scope(name):
        
        out = x
        context = spatial_pool(x, channel)
        channel_add_term = channel_add_conv(context, channel)
        out = out + channel_add_term
    
    return out
    
    

def res_blk(name, l, ch, ksize, count, split=1, strides=1, freeze=False, gcb=False):
    ch_in = l.get_shape().as_list()
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block' + str(i)):  
                x = l if i == 0 else BNReLU('preact', l)
                x = Conv2D('conv1', x, ch[0], ksize[0], activation=BNReLU)
                x = Conv2D('conv2', x, ch[1], ksize[1], split=split, 
                                strides=strides if i == 0 else 1, activation=BNReLU)
                x = Conv2D('conv3', x, ch[2], ksize[2], activation=tf.identity)
                if gcb:
                    x = context_block('gcb', x, ch[2])
                if (strides != 1 or ch_in[1] != ch[2]) and i == 0:
                    l = Conv2D('convshortcut', l, ch[2], 1, strides=strides)
                x = tf.stop_gradient(x) if freeze else x
                l = l + x
        # end of each group need an extra activation
        l = BNReLU('bnlast',l)  
    return l

def res_blk_basic(name, l, ch, ksize, count, split=1, strides=1, freeze=False, gcb=False):
    ch_in = l.get_shape().as_list()
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block' + str(i)):  
                x = l
                x = Conv2D('conv1', x, ch[0], ksize[0],
                        strides=strides if i == 0 else 1, activation=BNReLU)
                x = Conv2D('conv2', x, ch[1], ksize[1], activation=get_bn(True))
                if gcb:
                    x = context_block('gcb', x, ch[1])
                if strides != 1 and i == 0:
                    l = Conv2D('convshortcut', l, ch[1], 1, strides=strides, activation=get_bn(False))
                x = tf.stop_gradient(x) if freeze else x
                l = l + x 
    return tf.nn.relu(l)

def resnet_34_encoder(i, freeze, gcb):
    """
    ResNet34 Encoder
    """
    d1 = Conv2D('conv0',  i, 64, 7, padding='valid', strides=1, activation=BNReLU)
    d1 = res_blk_basic('group0', d1, [64, 64], [3, 3], 3, strides=1, freeze=freeze, gcb=gcb)
#     d1 = tf.stop_gradient(d1) if freeze else d1

    d2 = res_blk_basic('group1', d1, [128, 128], [3, 3], 4, strides=2, freeze=freeze, gcb=gcb)
    d2 = tf.stop_gradient(d2) if freeze else d2

    d3 = res_blk_basic('group2', d2, [256, 256], [3, 3], 6, strides=2, freeze=freeze, gcb=gcb)
    d3 = tf.stop_gradient(d3) if freeze else d3

    d4 = res_blk_basic('group3', d3, [512, 512], [3, 3], 3, strides=2, freeze=freeze, gcb=gcb)
    d4 = tf.stop_gradient(d4) if freeze else d4

    #d4 = Conv2D('conv_bot',  d4, 1024, 1, padding='same')
    return d1, d2, d3, d4

def resnet_50_preact(i, freeze, gcb):
    """
    Pre-activated ResNet50 Encoder
    """
    d1 = Conv2D('conv0',  i, 64, 7, padding='valid', strides=1, activation=BNReLU)
    d1 = res_blk('group0', d1, [ 64,  64,  256], [1, 3, 1], 3, strides=1, freeze=freeze, gcb=gcb)                       

    d2 = res_blk('group1', d1, [128, 128,  512], [1, 3, 1], 4, strides=2, freeze=freeze, gcb=gcb)
    d2 = tf.stop_gradient(d2) if freeze else d2

    d3 = res_blk('group2', d2, [256, 256, 1024], [1, 3, 1], 6, strides=2, freeze=freeze, gcb=gcb)
    d3 = tf.stop_gradient(d3) if freeze else d3

    d4 = res_blk('group3', d3, [512, 512, 2048], [1, 3, 1], 3, strides=2, freeze=freeze, gcb=gcb)
    d4 = tf.stop_gradient(d4) if freeze else d4

    d4 = Conv2D('conv_bot',  d4, 1024, 1, padding='same')
    return d1, d2, d3, d4

def encoder_blk(name, feat_in, num_feats, has_down=[False,False], gcb=False):
    with tf.variable_scope(name):
        feat = Conv2D('conv_1', feat_in, num_feats, 3, padding='same', strides=2 if has_down[0] else 1, activation=BNReLU)
        feat = Conv2D('conv_2', feat,   num_feats, 3, padding='same', strides=2 if has_down[1] else 1, activation=BNReLU)
        if gcb:
            feat = context_block('gcb', feat, num_feats)
        return feat    

def decoder_blk(name, feat_in, num_feats, shorcut):
    with tf.variable_scope(name):
        in_ch = feat_in.get_shape().as_list()[1]
#                 feat = Conv2DTranspose('us', feat_in, in_ch, 2, strides=(2, 2), padding='same', activation=tf.identity)
        feat = upsample2x('us', feat_in)
        feat = tf.concat([feat, shorcut], axis=1)
        feat = Conv2D('conv_1', feat, num_feats, 3, padding='same', strides=1, activation=BNReLU)
        feat = Conv2D('conv_2', feat, num_feats, 3, padding='same', strides=1, activation=BNReLU)
        return feat

def head_fusion(d1, d2, d3, name=""):
    with tf.variable_scope("head_fusion" + name):
        d = d1
        if d2 is not None:
            d2_up = upsample2x('up_0', d2)
            d = tf.concat([d, d2_up], 1)
        if d3 is not None:
            d3_up = upsample2x('up_1', d3)
            d3_up = upsample2x('up_2', d3_up)
            d = tf.concat([d, d3_up], 1)
        d = Conv2D('conv_mix', d, 256, 1, strides=1, activation=BNReLU)
    return d

def lead_encoder(i, freeze, gcb, name, img_fea=None):
    
    def composite_connection(name, f, ch, times=1):
        with tf.variable_scope(name):
            d = upsample2x('up_0', f)
            for idx in range(times-1):
                d = upsample2x('up_%s' % str(idx+1), d)
            d = Conv2D('conv_up', d, ch, 1, strides=1, activation=BNReLU)
        return d
    
    with tf.variable_scope(name):
        
        d11 = tf.add_n([i[-3], composite_connection('com_11', i[-2], 128, 1)])
        if img_fea is not None:
#             img_fea_1 = encoder_blk('conv_img11', img_fea, 128, has_down=True)
            img_fea_1 = img_fea[0]
            d11 = tf.concat([d11, img_fea_1], 1)
            d11 = Conv2D('conv_img12', d11, 128, 1, strides=1, activation=BNReLU)

        d11_res = res_blk_basic('groupl11', d11, [128, 128], [3, 3], 1, strides=1, freeze=freeze, gcb=gcb)

        d12 = Conv2D('conv_straight11', d11_res, 128, 3, strides=1, activation=BNReLU)

        d22 = Conv2D('conv_down11', d11_res, 256, 3, strides=2, activation=BNReLU)

        d12_sum = tf.add_n([d12, composite_connection('com_12',i[-1], 128, 2)])

        d22_sum = tf.add_n([d22, composite_connection('com_22',i[-1], 256, 1)])
        if img_fea is not None:
#             img_fea_2 = encoder_blk('conv_img21', img_fea_1, 256, has_down=True)
            img_fea_2 = img_fea[1]
            d22_sum = tf.concat([d22_sum, img_fea_2], 1)
            d22_sum = Conv2D('conv_img22', d22_sum, 256, 1, strides=1, activation=BNReLU)

        d12_res = res_blk_basic('groupl12', d12_sum, [128, 128], [3, 3], 1, strides=1, freeze=freeze, gcb=gcb)
        
        d22_res = res_blk_basic('groupl22', d22_sum, [256, 256], [3, 3], 1, strides=1, freeze=freeze, gcb=gcb)

        d13 = Conv2D('conv_straight12', d12_res, 128, 3, strides=1, activation=BNReLU)
        
        d13_sum = tf.add_n([d13, composite_connection('com_13', d22_res, 128, 1)])

        d23 = Conv2D('conv_straight22', d22_res, 256, 3, strides=1, activation=BNReLU)
        
        d12_down = Conv2D('conv_down12', d12_res, 256, 3, strides=2, activation=BNReLU)
        
        d23_sum = tf.add_n([d23, d12_down])

        d22_down = Conv2D('conv_down22', d22_res, 512, 3, strides=2, activation=BNReLU)
        
        
        d12_downdown = Conv2D('conv_downdown12', d12_down, 512, 3, strides=2, activation=BNReLU)
        
        d33_sum = tf.add_n([d22_down, d12_downdown])
        if img_fea is not None:
#             img_fea_3 = encoder_blk('conv_img31', img_fea_2, 512, has_down=True)
            img_fea_3 = img_fea[2]
            d33_sum = tf.concat([d33_sum, img_fea_3], 1)
            d33_sum = Conv2D('conv_img32', d33_sum, 512, 1, strides=1, activation=BNReLU)

        d13_res = res_blk_basic('groupl13', d13_sum, [128, 128], [3, 3], 1, strides=1, freeze=freeze, gcb=gcb)

        d23_res = res_blk_basic('groupl23', d23_sum, [256, 256], [3, 3], 1, strides=1, freeze=freeze, gcb=gcb)

        d33_res = res_blk_basic('groupl33', d33_sum, [512, 512], [3, 3], 1, strides=1, freeze=freeze, gcb=gcb)

        d14 = Conv2D('conv_straight13', d13_res, 128, 3, strides=1, activation=BNReLU)
        
        d14_sum = tf.add_n([d14, composite_connection('com_141', d23_res, 128, 1)])
        
        d14_sum = tf.add_n([d14_sum, composite_connection('com_142',d33_res, 128, 2)])
        

        d24 = Conv2D('conv_straight23', d23_res, 256, 3, strides=1, activation=BNReLU)
        
        d13_down = Conv2D('conv_down13', d13_res, 256, 3, strides=2, activation=BNReLU)
        
        d24_sum = tf.add_n([d24, d13_down])
        d24_sum = tf.add_n([d24_sum, composite_connection('com_24', d33_res, 256, 1)])

        d34 = Conv2D('conv_straight33', d33_res, 512, 3, strides=1, activation=BNReLU)
        
        d13_downdown = Conv2D('conv_downdown13', d13_down, 512, 3, strides=2, activation=BNReLU)
        
        d23_down = Conv2D('conv_down23', d23_res, 512, 3, strides=2, activation=BNReLU)
        
        d34_sum = tf.add_n([d34, d13_downdown])
        d34_sum = tf.add_n([d34_sum, d23_down])

    return [d11, d12_sum, d13_sum, d22_sum, d23_sum, d33_sum], [d14_sum, d24_sum, d34_sum]

def final_out_block(name, x, channels):
    with tf.variable_scope(name):
        feat = upsample2x('up', x)
        logi = Conv2D('conv_out', feat, channels, 1, use_bias=True, activation=tf.identity)
    return logi
    
    
class Graph(ModelDesc, Config):
    def __init__(self, freeze=False, training_strategy=True):
        super(Graph, self).__init__()
        assert tf.test.is_gpu_available()
        self.freeze = freeze
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
        
        # 40x image
        crop_imgs = crop_op(orig_imgs, (6, 6), "channels_last")
        # 10x image
        sub_imgs = tf.image.resize_bicubic(crop_imgs, [int(self.train_mask_shape[0]/4), int(self.train_mask_shape[1]/4)])

        pen_map = truemap_coded[...,-1]
        positive_map = truemap_coded[...,-2:-1]/255
        
        # classificaition map
        true = truemap_coded[...,1]           
        true = tf.cast(true, tf.int32)
        true = tf.identity(true, name='truemap')
        one  = tf.one_hot(true, self.nr_types, axis=-1)
        true = tf.expand_dims(true, axis=-1)  
        
        # binary map
        one_binary = tf.concat([one[...,:1], tf.expand_dims(tf.reduce_sum(tf.gather(one, [1,2,3,4], axis=-1), -1),
                         axis=-1)], -1, name='binary_labels_one_hot')
            
        G1v23_labels = tf.concat([one[...,:2], tf.expand_dims(tf.reduce_sum(tf.gather(one, [2,3], axis=-1), -1),
                         axis=-1), one[...,4:]], -1, name='G1v23_labels_one_hot')
        G12v3_labels = tf.concat([one[...,:1], tf.expand_dims(tf.reduce_sum(tf.gather(one, [1,2], axis=-1), -1),
                         axis=-1), one[...,3:]], -1, name='G12v3_labels_one_hot')
                     
        #### Xavier initializer
        with argscope(Conv2D, activation=tf.identity, use_bias=False,
                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()), \
                argscope([Conv2D, BatchNorm, MaxPooling, Conv2DTranspose, LayerNorm], data_format=self.data_format):

            i = tf.transpose(images, [0, 3, 1, 2])
            i = i if not self.input_norm else i / 255.0
            
            if self.mix_class:
                i_10x = tf.transpose(sub_imgs, [0, 3, 1, 2])
                i_10x = i_10x if not self.input_norm else i_10x / 255.0
                i_40x = tf.transpose(crop_imgs, [0, 3, 1, 2])
                i_40x = i_40x if not self.input_norm else i_40x / 255.0

                imgs_fea_10x_2 = encoder_blk('en_10x', i_10x, 128, has_down=[False,False], gcb=True)
#                 imgs_fea_10x_2 = res_blk_basic('en_10x', i_10x, [128, 128], [3, 3], 3, strides=1, freeze=False, gcb=self.gcb)
                imgs_fea_10x_1 = upsample2x('up_10x', imgs_fea_10x_2)
                imgs_fea_10x_1 = Conv2D('cov_10x_up', imgs_fea_10x_1, 64, 3, strides=1, activation=BNReLU)
                imgs_fea_10x_3 = Conv2D('cov_10x_down', imgs_fea_10x_2, 256, 3, strides=2, activation=BNReLU)
                imgs_fea_10x_2 = Conv2D('cov_10x_straight', imgs_fea_10x_2, 128, 3, strides=1, activation=BNReLU)
            
                imgs_fea_10x = [imgs_fea_10x_1, imgs_fea_10x_2, imgs_fea_10x_3]
                
                imgs_fea_40x_2 = encoder_blk('en_40x', i_40x, 128, has_down=[True,True], gcb=True)
#                 imgs_fea_40x_2 = res_blk_basic('en_40x', i_40x, [128, 128], [3, 3], 3, strides=2, freeze=False, gcb=self.gcb)
                imgs_fea_40x_1 = upsample2x('up_40x', imgs_fea_40x_2)
                imgs_fea_40x_1 = Conv2D('cov_40x_up', imgs_fea_40x_1, 64, 3, strides=1, activation=BNReLU)
                imgs_fea_40x_3 = Conv2D('cov_40x_down', imgs_fea_40x_2, 256, 3, strides=2, activation=BNReLU)
                imgs_fea_40x_2 = Conv2D('cov_40x_straight', imgs_fea_40x_2, 128, 3, strides=1, activation=BNReLU)
            
                imgs_fea_40x = [imgs_fea_40x_1, imgs_fea_40x_2, imgs_fea_40x_3]
    
            d = resnet_34_encoder(i, freeze=self.freeze, gcb=self.gcb)
            
            if self.type_classification:
            #### classification branch 12 vs 3
            # 64x64 -> strides 1 -> 64x64 -> upsample -> 128x128
                if self.mix_class:
                    s_group_1, s_head_1 = lead_encoder(d, freeze=False, gcb=self.gcb, name='classify_1', img_fea=imgs_fea_10x)
                    s_1 = head_fusion(s_head_1[0], s_head_1[1], s_head_1[2], name='_classify_1')
                    s_1_final = final_out_block("classify_1", s_1, self.nr_types-1)
                    logi_1 = tf.transpose(s_1_final, [0, 2, 3, 1])
                    soft_1 = tf.nn.softmax(logi_1, axis=-1)

                    #### classification branch 1 vs 23
                    s_group_2, s_head_2 = lead_encoder(d, freeze=False, gcb=self.gcb, name='classify_2', img_fea=imgs_fea_40x)
                    s_2 = head_fusion(s_head_2[0], s_head_2[1], s_head_2[2], name='_classify_2')
                    s_2_final = final_out_block("classify_2", s_2, self.nr_types-1)
                    logi_2 = tf.transpose(s_2_final, [0, 2, 3, 1])
                    soft_2 = tf.nn.softmax(logi_2, axis=-1)

                    ### fusion block
                    s = tf.concat([s_1_final, s_2_final], 1)
                    logi = Conv2D('conv_out_type', s, self.nr_types, 1, use_bias=True, activation=tf.identity)
                    logi = tf.transpose(logi, [0, 2, 3, 1])
                    soft = tf.nn.softmax(logi, axis=-1)
                else:
                    s_group, s_head = lead_encoder(d, freeze=False, gcb=self.gcb, name='classify', img_fea=None)
                    s = head_fusion(s_head[0], s_head[1], s_head[2], name='_classify')
                    s = final_out_block("classify", s, self.nr_types)
                    logi = tf.transpose(s, [0, 2, 3, 1])
                    soft = tf.nn.softmax(logi, axis=-1)
            
            if self.type_nuclei:
                ### binary branch
                s_binary = decoder_blk('u3b',  d[-1], 256, d[-2])
                s_binary = decoder_blk('u2b', s_binary, 128, d[-3])
                s_binary = decoder_blk('u1b', s_binary, 64, d[-4])
                logi_binary = Conv2D('conv_out_b', s_binary, self.nr_classes, 
                            1, use_bias=True, activation=tf.identity)
                logi_binary = tf.transpose(logi_binary, [0, 2, 3, 1])
                soft_binary = tf.nn.softmax(logi_binary, axis=-1)
                
            if self.regression:
                s_reg_1 = encoder_blk('d1r', tf.transpose(logi_binary, [0, 3, 1, 2]), 64, has_down=[False, False]) # 256
                s_reg_2 = encoder_blk('d2r', s_reg_1, 128, has_down=[True, False]) # 128
                s_reg_3 = encoder_blk('d3r', s_reg_2, 256, has_down=[True, False]) # 64
                s_reg = decoder_blk('u2r',  s_reg_3, 128, s_reg_2) # 64 -> 128
                s_reg = decoder_blk('u1r', s_reg, 64, s_reg_1) # 128 -> 256
#                 s_reg = decoder_blk('u3b',  d[-1], 256, d[-2])
#                 s_reg = decoder_blk('u2b', s_reg, 128, d[-3])
#                 s_reg = decoder_blk('u1b', s_reg, 64, d[-4])
                soft_reg = Conv2D('conv_out_r', s_reg, 1, 
                            1, use_bias=True, activation=tf.sigmoid)
                soft_reg = tf.transpose(soft_reg, [0, 2, 3, 1])
            
            if self.type_nuclei:
                prob_np = tf.identity(soft_binary[...,1], name='predmap-prob')
                prob_np = tf.expand_dims(prob_np, axis=-1)
            else:
                prob_np = tf.reduce_sum(soft[...,1:], axis=-1, keepdims=True)
                prob_np = tf.identity(prob_np, name='predmap-prob')
                
            if self.type_classification:
                predmap_coded = tf.concat([soft, prob_np], axis=-1)
            else:
                predmap_coded = prob_np
                
            if self.regression:
                predmap_coded = tf.concat([predmap_coded, soft_reg], axis=-1)
#                 predmap_coded = soft_reg
              
            # * channel ordering: type-map, segmentation map
            # encoded so that inference can extract all output at once
            predmap_coded = tf.identity(predmap_coded, name='predmap-coded')
            
        def mae_loss(true, pred, name=None):
            ### regression loss
            loss = tf.abs(pred - true)
            return loss
        
        a1 = tf.get_variable("param1", [1], initializer = tf.constant_initializer(1.))
        a2 = tf.get_variable("param2", [1], initializer = tf.constant_initializer(1.))
        a3 = tf.get_variable("param3", [1], initializer = tf.constant_initializer(1.))
        a4 = tf.get_variable("param4", [1], initializer = tf.constant_initializer(1.))
        a5 = tf.get_variable("param5", [1], initializer = tf.constant_initializer(1.))
        
        ####
        if is_training:
            ######## LOSS
            self.cost = 0
            
            if self.type_classification:
                ### classification 1 loss
                if self.mix_class:
                    loss_ce_1 = categorical_crossentropy(soft_1, G12v3_labels)
                    loss_ce_1 = tf.reduce_mean(loss_ce_1, name='loss-ce-12v3')
                    add_moving_summary(loss_ce_1)

                    loss_dice_1 = 0
                    for type_id in range(self.nr_types-1):
                        loss_dice_1 += dice_loss(soft_1[...,type_id], 
                                           G12v3_labels[...,type_id])
                    loss_dice_1 = tf.identity(loss_dice_1, name='loss-dice-12v3')
                    add_moving_summary(loss_dice_1)

                    ### classification 2 loss
                    loss_ce_2 = categorical_crossentropy(soft_2, G1v23_labels)
                    loss_ce_2 = tf.reduce_mean(loss_ce_2, name='loss-ce-1v23')
                    add_moving_summary(loss_ce_2)

                    loss_dice_2 = 0
                    for type_id in range(self.nr_types-1):
                        loss_dice_2 += dice_loss(soft_2[...,type_id], 
                                           G1v23_labels[...,type_id])
                    loss_dice_2 = tf.identity(loss_dice_2, name='loss-dice-1v23')
                    add_moving_summary(loss_dice_2)

                ### final classification loss
                coe = tf.constant([1.0,1.0,5.0,5.0,1.0])
                one_coe = one*coe
                loss_ce = categorical_crossentropy(soft, one)
                loss_ce = tf.reduce_mean(loss_ce, name='loss-ce')
                add_moving_summary(loss_ce)
                
                loss_dice = 0
                for type_id in range(self.nr_types):
                    loss_dice += dice_loss(soft[...,type_id], one[...,type_id]) #* coe[type_id]
                loss_dice = tf.identity(loss_dice, name='loss-dice')
                add_moving_summary(loss_dice)
                
                if self.mix_class:
                    if self.use_dice:
                        loss_classify = loss_ce + loss_ce_1 + loss_ce_2 + loss_dice_1 + loss_dice_2 + loss_dice
                    else:
                        loss_classify = loss_ce + loss_ce_1 + loss_ce_2
                else:
                    if self.use_dice:
                        loss_classify = loss_ce + loss_dice
                    else:
                        loss_classify = loss_ce
                
                if self.uncertainty:
                    loss_classify = tf.reduce_sum(1/(a1*a1)*loss_classify + tf.log(a1))
                    a1_v = tf.reduce_sum(a1, name='param_classify')
                    add_moving_summary(a1_v)
                self.cost += loss_classify
                
                
            if self.type_nuclei:
                ### binary loss
                loss_bce = categorical_crossentropy(soft_binary, one_binary)
                loss_bce = tf.reduce_mean(loss_bce, name='loss-bce')
                add_moving_summary(loss_bce)

                loss_dice_b = 0
                for type_id in range(self.nr_classes):
                    loss_dice_b += dice_loss(soft_binary[...,type_id], 
                                       one_binary[...,type_id])
                loss_dice_b = tf.identity(loss_dice_b, name='loss-dice-binary')
                add_moving_summary(loss_dice_b)
                if self.use_dice:
                    loss_binary = loss_bce + loss_dice_b
                else:
                    loss_binary = loss_bce
                    
                if self.uncertainty:
                    loss_binary = tf.reduce_sum(1/(a4*a4)*loss_binary + tf.log(a4))
                    a4_v = tf.reduce_sum(a4, name='param_binary')
                    add_moving_summary(a4_v)
                    
                self.cost += loss_binary
                   
            if self.regression:
                ## regression
                loss_reg = mae_loss(positive_map, soft_reg)
                loss_reg = tf.reduce_mean(loss_reg, name='loss-reg-mae')
                add_moving_summary(loss_reg)
                
                if self.uncertainty:
                    loss_reg = tf.reduce_sum(1/(a5*a5)*loss_reg + tf.log(a5))
                    a5_v = tf.reduce_sum(a5, name='param_regression')
                    add_moving_summary(a5_v)
                
                self.cost += (loss_reg)
                
#             add_moving_summary(('.param*'))
            add_param_summary(('.*/W', ['histogram']), ('.*/param*', ['scalar']))   # monitor W

        return
