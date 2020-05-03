# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         MAIN                                          | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

from unet import unet
from data import *

from model import Model
from trainer import Trainer
from tensorflow.keras.callbacks import ModelCheckpoint

from config import dataloader_config, model_config, train_config

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

##### Building the train and test generator #####

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

train_generator = trainGenerator(dataloader_config['batch_size'], dataloader_config['train_path'], dataloader_config['image_folder'], 
                        dataloader_config['mask_folder'], data_gen_args, "grayscale", dataloader_config['save_to_dir'])
test_generator = testGenerator(dataloader_config['test_path'])


##### Building the Model and Trainer class from the config script #####
# A Model is composed of a network, an optimizer, a scheduler and a loss
# We then use a Trainer to train or test a net from a Model 

model = Model(model_config, dataloader_config)
trainer = Trainer(model, dataloader_config, train_config, train_generator, test_generator)
print(trainer)


##### Get the model checkpoint from a .hdf5 file #####

model_checkpoint = ModelCheckpoint(train_config['checkpoint'], monitor='loss',verbose=1, save_best_only=True)


##### Train and/or predict from the Trainer #####

if train_config['mode'] == 'train and predict':
    trainer.train(model_checkpoint)
    results = trainer.predict()
    saveResult(dataloader_config['test_path'], results) # Saving the results to the test_path

elif train_config['mode'] == 'train':
    trainer.train(model_checkpoint)

elif train_config['mode'] == 'predict':
    results = trainer.predict()
    saveResult(dataloader_config['test_path'], results) # Saving the results to the test_path
    