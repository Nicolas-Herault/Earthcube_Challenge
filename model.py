import math
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.losses import Reduction 
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

from unet import *

from config import model_config


""" 
This class is basically nothing but instanciation. 
Methods from model.py should never be called. 
They are either called during instanciation or by method from a Trainer object.
"""

class Model():
    
    def __init__(self, model_config, dataloader_config, pretrained_weights=None):
        self.loss  = self._init_loss(model_config)
        self.optimizer  = self._init_optimizer()
        self.scheduler  = self._init_scheduler()
        self.net        = self._init_net(model_config, pretrained_weights)
        
    
    
    def _init_net(self, model_config, pretrained_weights):
        if model_config['net']=='unet':
            net = unet(pretrained_weights)
            print('Network build !')
            net.compile(optimizer = self.optimizer, loss = self.loss, metrics = ['accuracy'])
            print('Network compiled !')
        return net

        
                
    def _init_loss(self, model_config):
        if model_config['loss'] == 'binary_crossentropy' :
            loss = 'binary_crossentropy'
        return loss
        
    
    def _init_optimizer(self):
        if model_config['optimizer'] == 'SGD':
            optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        elif model_config['optimizer'] == 'Adam':
            optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                amsgrad=False,name='Adam')
        return optimizer
    
    #Even if scheduler are callbacks define in the training, it's specific to a model
    #This is why we choose to define the scheduler in the Model class
    def _init_scheduler(self):
        if model_config['scheduler'] == 'ROP':
            scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=0, 
                                        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        return scheduler
