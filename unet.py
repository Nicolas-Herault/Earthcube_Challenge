from config import arch_config
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np 
import tensorflow
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from tensorflow.keras import backend as keras

from block import classicBlock, resnetBlock

""" 
This class is the architecture of the Unet.
It takes many parameters from the config script, so if you want to modify the architecture, you should do it in the config.
If something as not yet being implemented in the architecture, then I recommend that you add it below and then add hyperparameters in the config script.
Therefore, you can have an overall view of all your code in the config script !
"""


########################## ARCHITECTURE CONSTRUCTION ########################################################


def unet(pretrained_weights = None, input_size = (256,256,1)):
    
    conv_list = []
    inputs = Input(input_size)
    
    out = inputs

    for pooling_index in range (arch_config['num_pooling'] - 1):
        out = PoolBlock(out, pooling_index, arch_config['train_padding'], arch_config['block_type'],
                        arch_config['num_layers_before_pooling'], arch_config['batch_normalization'], conv_list)

    out = ExtensionBlock(out, arch_config['train_padding'], arch_config['block_type'],
                        arch_config['num_layers_before_pooling'], arch_config['batch_normalization'])

    for up_index in range(arch_config['num_pooling'] - 1):
        out = UpBlock(out, up_index, arch_config['train_padding'], arch_config['block_type'],
                        arch_config['num_layers_before_pooling'], arch_config['batch_normalization'], conv_list)

    out = Conv2D(2, 3, activation = 'relu', padding = arch_config['train_padding'], kernel_initializer = 'he_normal')(out)
    out = Conv2D(1, 1, activation = 'sigmoid')(out)

    model = tensorflow.keras.Model(inputs=inputs, outputs=out)

    if pretrained_weights is not None :
        model.load_weights(self.pretrained_weights)

    return model


########################## UNET CONSTRUCTION FUNCTION  WITH BLOCK #########################################


def PoolBlock(inputs, pooling_index, padding, block, num_layers_before_pooling, batch_normalization, conv_list):
    
    out = inputs 
    
    if block == 'classic':
        out = classicBlock(inputs = out, size = 64*(2**pooling_index), padding = padding,
                        num_layers_before_pooling = num_layers_before_pooling, batch_normalization = batch_normalization).call()
    elif block == 'resnet':
        out = resnetBlock(inputs = out, size = 64*(2**pooling_index), padding = padding,
                        num_layers_before_pooling = num_layers_before_pooling).call()

    conv_list.append(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    
    return out

def UpBlock(out, up_index, padding, block, num_layers_before_pooling, batch_normalization, conv_list):
    
    num_pooling = arch_config['num_pooling']

    up = Conv2D(64*(2**(num_pooling - up_index)), 2, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(out))
    conv = conv_list[-(up_index + 1)]
    merge = concatenate([conv,up], axis = 3)

    if block == 'classic':
        out = classicBlock(inputs = merge, size = 64*(2**(num_pooling - up_index)), padding = padding,
                        num_layers_before_pooling = num_layers_before_pooling, batch_normalization = batch_normalization).call()
    elif block == 'resnet':
        out = resnetBlock(inputs = merge, size = 64*(2**(num_pooling - up_index)), padding = padding,
                        num_layers_before_pooling = num_layers_before_pooling).call()

    return out
        
def ExtensionBlock(inputs, padding, block, num_layers_before_pooling, batch_normalization):
    
    num_pooling = arch_config['num_pooling']
    
    out = inputs
    
    if block == 'classic':
        out = classicBlock(inputs = out, size = 64*(2**num_pooling), padding = padding,
                        num_layers_before_pooling = num_layers_before_pooling, batch_normalization = batch_normalization).call()
    elif block == 'resnet':
        out = resnetBlock(inputs = out, size = 64*(2**num_pooling), padding = padding,
                        num_layers_before_pooling = num_layers_before_pooling).call()

    drop1 = Dropout(0.5)(out)
    out = MaxPooling2D(pool_size=(2, 2))(drop1)

    if block == 'classic':
        out = classicBlock(inputs = out, size = 64*(2**(num_pooling + 1)), padding = padding,
                        num_layers_before_pooling = num_layers_before_pooling, batch_normalization = batch_normalization).call()
    elif block == 'resnet':
        out = resnetBlock(inputs = out, size = 64*(2**(num_pooling + 1)), padding = padding,
                        num_layers_before_pooling = num_layers_before_pooling).call()


    drop2 = Dropout(0.5)(out)
    up = Conv2D(64*(2**num_pooling), 2, activation = 'relu', padding = padding, kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop2))
    merge = concatenate([drop1,up], axis = 3)

    if block == 'classic':
        out = classicBlock(inputs = merge, size = 64*(2**num_pooling), padding = padding,
                        num_layers_before_pooling = num_layers_before_pooling, batch_normalization = batch_normalization).call()
    elif block == 'resnet':
        out = resnetBlock(inputs = merge, size = 64*(2**num_pooling), padding = padding,
                        num_layers_before_pooling = num_layers_before_pooling).call()

    return out
