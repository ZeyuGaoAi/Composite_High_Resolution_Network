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

def lead_encoder(i, freeze, gcb, name):
    
    def composite_connection(name, f, ch, times=1):
        with tf.variable_scope(name):
            d = upsample2x('up_0', f)
            for idx in range(times-1):
                d = upsample2x('up_%s' % str(idx+1), d)
            d = Conv2D('conv_up', d, ch, 1, strides=1, activation=BNReLU)
        return d
    
    with tf.variable_scope(name):
        
        d11 = tf.add_n([i[-3], composite_connection('com_11', i[-2], 128, 1)])

        d11_res = res_blk_basic('groupl11', d11, [128, 128], [3, 3], 1, strides=1, freeze=freeze, gcb=gcb)
        d11_res = tf.stop_gradient(d11_res) if freeze else d11_res

    #     d11 = tf.add_n([i[-3], composite_connection('com_11', i[-2], 512, 1)])

    #     d11 = res_blk('groupl11', d11, [32, 32, 128], [1, 3, 1], 1, strides=1, freeze=freeze, gcb=gcb)

        d12 = Conv2D('conv_straight11', d11_res, 128, 3, strides=1, activation=BNReLU)
        d12 = tf.stop_gradient(d12) if freeze else d12

        d22 = Conv2D('conv_down11', d11_res, 256, 3, strides=2, activation=BNReLU)
        d22 = tf.stop_gradient(d22) if freeze else d22

        d12_sum = tf.add_n([d12, composite_connection('com_12',i[-1], 128, 2)])

        d22_sum = tf.add_n([d22, composite_connection('com_22',i[-1], 256, 1)])

        d12_res = res_blk_basic('groupl12', d12_sum, [128, 128], [3, 3], 1, strides=1, freeze=freeze, gcb=gcb)
        d12_res = tf.stop_gradient(d12_res) if freeze else d12_res
        
        d22_res = res_blk_basic('groupl22', d22_sum, [256, 256], [3, 3], 1, strides=1, freeze=freeze, gcb=gcb)
        d22_res = tf.stop_gradient(d22_res) if freeze else d22_res

    #     d12_sum = res_blk('groupl12', d12_sum, [32, 32, 128], [1, 3, 1], 1, strides=1, freeze=freeze, gcb=gcb)

    #     d22_sum = res_blk('groupl22', d22_sum, [64, 64, 256], [1, 3, 1], 1, strides=1, freeze=freeze, gcb=gcb)

        d13 = Conv2D('conv_straight12', d12_res, 128, 3, strides=1, activation=BNReLU)
        d13 = tf.stop_gradient(d13) if freeze else d13
        
        d13_sum = tf.add_n([d13, composite_connection('com_13', d22_res, 128, 1)])
        d13_sum = tf.stop_gradient(d13_sum) if freeze else d13_sum

        d23 = Conv2D('conv_straight22', d22_res, 256, 3, strides=1, activation=BNReLU)
        d23 = tf.stop_gradient(d23) if freeze else d23
        
        d12_down = Conv2D('conv_down12', d12_res, 256, 3, strides=2, activation=BNReLU)
        d12_down = tf.stop_gradient(d12_down) if freeze else d12_down
        
        d23_sum = tf.add_n([d23, d12_down])

        d22_down = Conv2D('conv_down22', d22_res, 512, 3, strides=2, activation=BNReLU)
        d22_down = tf.stop_gradient(d22_down) if freeze else d22_down
        
        
        d12_downdown = Conv2D('conv_downdown12', d12_down, 512, 3, strides=2, activation=BNReLU)
        d12_downdown = tf.stop_gradient(d12_downdown) if freeze else d12_downdown
        
        d33_sum = tf.add_n([d22_down, d12_downdown])

        d13_res = res_blk_basic('groupl13', d13_sum, [128, 128], [3, 3], 1, strides=1, freeze=freeze, gcb=gcb)
        d13_res = tf.stop_gradient(d13_res) if freeze else d13_res

        d23_res = res_blk_basic('groupl23', d23_sum, [256, 256], [3, 3], 1, strides=1, freeze=freeze, gcb=gcb)
        d23_res = tf.stop_gradient(d23_res) if freeze else d23_res

        d33_res = res_blk_basic('groupl33', d33_sum, [512, 512], [3, 3], 1, strides=1, freeze=freeze, gcb=gcb)
        d33_res = tf.stop_gradient(d33_res) if freeze else d33_res

    #     d13_sum = res_blk('groupl13', d13_sum, [32, 32, 128], [1, 3, 1], 1, strides=1, freeze=freeze, gcb=gcb)

    #     d23_sum = res_blk('groupl23', d23_sum, [64, 64, 256], [1, 3, 1], 1, strides=1, freeze=freeze, gcb=gcb)

    #     d33_sum = res_blk('groupl33', d33_sum, [128, 128, 512], [1, 3, 1], 1, strides=1, freeze=freeze, gcb=gcb)

        d14 = Conv2D('conv_straight13', d13_res, 128, 3, strides=1, activation=BNReLU)
        d14 = tf.stop_gradient(d14) if freeze else d14
        
        d14_sum = tf.add_n([d14, composite_connection('com_141', d23_res, 128, 1)])
        
        d14_sum = tf.add_n([d14_sum, composite_connection('com_142',d33_res, 128, 2)])
        d14_sum = tf.stop_gradient(d14_sum) if freeze else d14_sum
        

        d24 = Conv2D('conv_straight23', d23_res, 256, 3, strides=1, activation=BNReLU)
        d24 = tf.stop_gradient(d24) if freeze else d24
        
        d13_down = Conv2D('conv_down13', d13_res, 256, 3, strides=2, activation=BNReLU)
        d13_down = tf.stop_gradient(d13_down) if freeze else d13_down
        
        d24_sum = tf.add_n([d24, d13_down])
        d24_sum = tf.add_n([d24_sum, composite_connection('com_24', d33_res, 256, 1)])
        d24_sum = tf.stop_gradient(d24_sum) if freeze else d24_sum

        d34 = Conv2D('conv_straight33', d33_res, 512, 3, strides=1, activation=BNReLU)
        d34 = tf.stop_gradient(d34) if freeze else d34
        
        d13_downdown = Conv2D('conv_downdown13', d13_down, 512, 3, strides=2, activation=BNReLU)
        d13_downdown = tf.stop_gradient(d13_downdown) if freeze else d13_downdown
        
        d23_down = Conv2D('conv_down23', d23_res, 512, 3, strides=2, activation=BNReLU)
        d23_down = tf.stop_gradient(d23_down) if freeze else d23_down
        
        d34_sum = tf.add_n([d34, d13_downdown])
        d34_sum = tf.add_n([d34_sum, d23_down])
        d34_sum = tf.stop_gradient(d34_sum) if freeze else d34_sum

    return [d11, d12_sum, d13_sum, d22_sum, d23_sum, d33_sum], [d14_sum, d24_sum, d34_sum]

def final_out_block(name, x, channels):
    with tf.variable_scope(name):
        feat = upsample2x('up', x)
        logi = Conv2D('conv_out', feat, channels, 1, use_bias=True, activation=tf.identity)
        logi = tf.transpose(logi, [0, 2, 3, 1])
        soft = tf.nn.softmax(logi, axis=-1)
    return soft
    
    
class Graph(ModelDesc, Config):
    def __init__(self, training_strategy=[True, True], freeze=False):
        super(Graph, self).__init__()
        assert tf.test.is_gpu_available()
        self.freeze = freeze
        self.training_strategy = training_strategy
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
        
#         sub_imgs = crop_op(images, (6, 6), "channels_last")

        pen_map = truemap_coded[...,-1]
        
        pos_map = truemap_coded[...,-3:-2]/255
        neg_map = truemap_coded[...,-2:-1]/255
        
        if hasattr(self, 'type_classification') and self.type_classification:
            true = truemap_coded[...,1]
        else:
            true = truemap_coded[...,0]            
        true = tf.cast(true, tf.int32)
        true = tf.identity(true, name='truemap')
        one  = tf.one_hot(true, self.nr_types if self.type_classification else self.nr_classes, axis=-1)
        
#         two_class_labels = tf.concat([one[...,:1], tf.expand_dims(tf.reduce_sum(one[...,1:], -1), axis=-1)],
#                                      -1, name='two_class_one_hot')
        
        if hasattr(self, 'auxilary_tasks') and self.auxilary_tasks:
            # bg vs cancer vs endo
            CvE_labels = tf.concat([one[...,:1], tf.expand_dims(tf.reduce_sum(one[...,1:4], -1), axis=-1), 
                                            one[...,4:]], -1, name='CvE_labels_one_hot')
            # bg vs 1 vs 23 vs endo
            G1v23_labels = tf.concat([one[...,:2], tf.expand_dims(tf.reduce_sum(tf.gather(one, [2,3], axis=-1), -1), axis=-1),
                             one[...,4:]], -1, name='G1v23_labels_one_hot')

            auxilary_labels = [CvE_labels, G1v23_labels, one]
        
        true = tf.expand_dims(true, axis=-1)               

        #### Xavier initializer
        with argscope(Conv2D, activation=tf.identity, use_bias=False,
                      kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()), \
                argscope([Conv2D, BatchNorm, MaxPooling, Conv2DTranspose, LayerNorm], data_format=self.data_format):

            i = tf.transpose(images, [0, 3, 1, 2])
            i = i if not self.input_norm else i / 255.0
            
            # classification branch

            d = resnet_34_encoder(i, freeze=self.freeze, gcb=self.gcb)
            
            if self.training_strategy[0]:
                freeze_classify = False
            else:
                freeze_classify = True
                
            s_group, s_head = lead_encoder(d, freeze=freeze_classify, gcb=self.gcb, name='classify')
            s = head_fusion(s_head[0], s_head[1], s_head[2], name='classify')
            
            if self.auxilary_tasks:
                with tf.variable_scope("auxilary"):
                    fea_1 = head_fusion(s_group[0], None, None, "_deep_sup_1")
                    soft_1 = final_out_block("deep_sup_1", fea_1, self.nr_types-2)
                    
                    fea_2 = head_fusion(s_group[1], s_group[3], None, "_deep_sup_2")
                    soft_2 = final_out_block("deep_sup_2", fea_2, self.nr_types-1)
                    
                    fea_3 = head_fusion(s_group[2], s_group[4], s_group[5], "_deep_sup_3")
                    soft_3 = final_out_block("deep_sup_3", fea_3, self.nr_types)
                    
                deep_soft = [soft_1, soft_2, soft_3]

            
            # final results 
            soft = final_out_block("final", s, self.nr_types)
            
            
            # regression branch
            
            if self.training_strategy[1]:
                freeze_2 = False
            else:
                freeze_2 = True
            
            if self.regression:
                
                _, reg_fea = lead_encoder(d, freeze=freeze_2, gcb=self.gcb, name='regression')
                pgx = head_fusion(reg_fea[0], reg_fea[1], reg_fea[2], name='pgx')
                ngx = head_fusion(reg_fea[0], reg_fea[1], reg_fea[2], name='ngx')
                #### Positive Guassian (PG)
                
                pgx = upsample2x('pgx_up', pgx)
                logi_pg = Conv2D('conv_out_pg', pgx, 1, 1, use_bias=True, activation=tf.sigmoid)
                logi_pg = tf.transpose(logi_pg, [0, 2, 3, 1])
                prob_pg = tf.identity(logi_pg, name='predmap-prob-pg')
                pred_pg = tf.identity(logi_pg, name='predmap-pg')
                
                #### Negative Guassian (NG)
#                 _, ngx = lead_encoder(d, freeze=freeze_2, gcb=self.gcb, name='ngreg')
                ngx = upsample2x('ngx_up', ngx)
                
                logi_ng = Conv2D('conv_out_ng', ngx, 1, 1, use_bias=True, activation=tf.sigmoid)
                logi_ng = tf.transpose(logi_ng, [0, 2, 3, 1])
                prob_ng = tf.identity(logi_ng, name='predmap-prob-ng')
                pred_ng = tf.identity(logi_ng, name='predmap-ng')
            
            
            if self.type_classification:
                prob_np = tf.reduce_sum(soft[...,1:], axis=-1, keepdims=True)
                prob_np = tf.identity(prob_np, name='predmap-prob-np')
                predmap_coded = tf.concat([soft, prob_np], axis=-1)
            if self.regression:
                predmap_coded = tf.concat([predmap_coded, pred_pg, pred_ng], axis=-1)
#             else:
#                 prob_np = tf.identity(soft[...,1], name='predmap-prob')
#                 prob_np = tf.expand_dims(prob_np, axis=-1)
#                 predmap_coded = tf.concat([prob_np, pred_pg, pred_ng], axis=-1)

            # * channel ordering: type-map, segmentation map
            # encoded so that inference can extract all output at once
            predmap_coded = tf.identity(predmap_coded, name='predmap-coded')
            
        def loss_mae(true, pred, name=None):
            ### regression loss
            loss = tf.abs(pred - true)
            loss = tf.reduce_mean(loss, name=name)
            return loss
        
        def loss_ssim(true, pred, name=None):
            ### region similarity loss
            loss = tf.image.ssim(tf.concat([true, true, true], -1),
                                 tf.concat([pred, pred, pred], -1), 1.0)
            loss = 1 - tf.reduce_mean(loss)
            loss = tf.identity(loss, name=name)
            return loss
        
        def loss_mcc(true, pred, name=None):
            ### robust loss
            loss = tf.exp(-tf.square(true-pred)/(0.2*0.2*2))
            loss = 1 - tf.reduce_mean(loss)
            loss = tf.identity(loss, name=name)
            return loss

        ####
        if is_training:
            ######## LOSS
            ### auxilary loss
            auxilary_loss = 0
            if self.auxilary_tasks:
                for i in range(len(deep_soft)):
                    loss_temp = categorical_crossentropy(deep_soft[i], auxilary_labels[i])
                    loss_temp = tf.reduce_mean(loss_temp)
                    auxilary_loss += loss_temp
                auxilary_loss = tf.identity(auxilary_loss, name='loss-auxilary')
                add_moving_summary(auxilary_loss)
            
            ### classification loss
            loss_bce = categorical_crossentropy(soft, one)
#             loss_bce = tf.reduce_mean(loss_bce * pen_map, name='loss-ce')
            loss_bce = tf.reduce_mean(loss_bce, name='loss-ce')
            add_moving_summary(loss_bce)
            
            loss_dice = 0
            for type_id in range(self.nr_types):
                loss_dice += dice_loss(soft[...,type_id], 
                                       one[...,type_id])
            loss_dice = tf.identity(loss_dice, name='loss-dice-class')
            add_moving_summary(loss_dice)
            
            ### regression loss
            if self.regression:
                loss_regress1 = loss_mae(pos_map, pred_pg) + loss_mae(neg_map, pred_ng)
                loss_regress2 = loss_ssim(pos_map, pred_pg) + loss_ssim(neg_map, pred_ng)
                loss_regress1 = tf.identity(loss_regress1, name='loss-mae')
                loss_regress2 = tf.identity(loss_regress2, name='loss-ssim')
                add_moving_summary(loss_regress1)
                add_moving_summary(loss_regress2)

#             wd_loss = regularize_cost('.*/W', l2_regularizer(1.0e-5), name='l2_wd_loss')
#             add_moving_summary(wd_loss)
            
            self.cost = 0
            
            if self.training_strategy[0]:
                self.cost += (loss_bce + auxilary_loss) # loss_dice
            if self.training_strategy[1]:
                self.cost += loss_regress1

            add_param_summary(('.*/W', ['histogram']))   # monitor W

            #### logging visual sthg
            orig_imgs = tf.cast(orig_imgs  , tf.uint8)
            orig_imgs = crop_op(orig_imgs, (6, 6), "channels_last")
            tf.summary.image('input', orig_imgs, max_outputs=1)
    
            pred = colorize(prob_np[...,0], cmap='jet')
            true = colorize(true[...,0], cmap='jet')
            pen_map = colorize(pen_map, cmap='jet')

            viz = tf.concat([orig_imgs, pred, true, pen_map], 2)

            tf.summary.image('output', viz, max_outputs=1)

        return
