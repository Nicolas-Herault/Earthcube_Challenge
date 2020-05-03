############################################ ARCHITECTURE ############################################

### This dictionary define the Unet ARCHITECTURE
### You can choose either choose the numbre of pooling operations and numbers of conv before one pool
### Other details of the architecture can be modified too

arch_config = {
    'num_pooling' : 4, # Define the number of pooling operations in your network
    'num_layers_before_pooling': 2, # Define the numbers of convolutions in your block before a pool
    'block_type' : 'standard', # Define the type of block, for now can be 'classic' or 'resnet'
    'train_padding': 'same',
    'predict_padding' : None, # Not implemented yet
    'batch_normalization': True,
}



############################################ MODEL ############################################

### The model dictionary define all the model
### By a model, we mean a network, an optimizer, a loss and a scheduler

model_config = {
    'net': 'unet',
    'optimizer': 'SGD', # can be 'SGD' or 'Adam'
    'loss' : 'binary_crossentropy', # As the loss is define by a string in a network.compile(), you can choose between many loss
    'scheduler': 'ROP', # Callbacks implemented but not functionnal yet
}

######################################### DATALOADER ##########################################

### Simple dictionary to manage the loading of data and the saving folder of trained models
### It is here that you can change the batch_size

dataloader_config = {
    'batch_size' : 2,
    'train_path' : 'data/membrane/train',
    'test_path' : 'data/membrane/test',
    'image_folder' : 'image',
    'mask_folder' : 'label',
    'save_to_dir' : 'checkpoints',
}

########################################### TRAINER #############################################

### The train dictionary will define all the training and predictions of models previously define
### You can also load weight from a previous checkpoint

train_config = {
    'nb_epochs' : 1,
    'steps_per_epoch' : 300,
    'train_verbose' : 1,
    'use_early_stopping': True, # En early stopping is a callback that stop the training if the network isn't learning anymore
    'patience': 30, # early stopping patience
    'delta': 0.01, # value of loss decrease to cancel early stopping
    'early_stopping_verbose': 1,
    'nb_samples_to_predict' : 30,
    'mode': 'train and predict', # Can be 'train', 'predict' or 'train and predict'
    'pretrained' : True,
    'checkpoint' : 'unet_membrane.hdf5',
}
 
