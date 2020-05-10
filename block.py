from tensorflow.keras.layers import *
from tensorflow.keras import layers


""" 
This class is just for definition of block. 
You can add any block that you want and then call them in the unet.py script.
"""

class classicBlock():

  def __init__(self, **kwargs):
    super(classicBlock, self).__init__()
    self.inputs = kwargs.get('inputs', None)
    self.size = kwargs.get('size', None)
    self.padding = kwargs.get('padding', None)
    self.batch_normalization = kwargs.get('batch_normalization', None)
    self.num_layers_before_pooling = kwargs.get('num_layers_before_pooling', None)

  def call(self):
    for i in range(self.num_layers_before_pooling - 1):
        x = Conv2D(self.size, 3, padding = self.padding, kernel_initializer = 'he_normal')(self.inputs)
        if self.batch_normalization :
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x

class resnetBlock():

  def __init__(self, **kwargs):
    super(resnetBlock, self).__init__()
    self.inputs = kwargs.get('inputs', None)
    self.size = kwargs.get('size', None)
    self.padding = kwargs.get('padding', None)
    self.num_layers_before_pooling = kwargs.get('num_layers_before_pooling', None)

  def call(self):
    for i in range(self.num_layers_before_pooling - 1):
        x = Conv2D(self.size, 3, padding = self.padding, kernel_initializer = 'he_normal')(self.inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
    x = Add(self.inputs, x)
    x = Activation('relu')(x)
    return x
