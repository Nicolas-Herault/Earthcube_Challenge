from config import train_config, model_config

from model import Model
from tensorflow.keras.callbacks import EarlyStopping

""" 
This class is for running train and predict on a specific model.
It is also made to define callbacks and for summaries of trainings/predictions.
"""

class Trainer():
     
    def __init__(self, model, dataloader_config, train_config, train_generator, test_generator):
        self.model = model # model must be an instance of the Model class
        self.train_generator, self.test_generator = train_generator, test_generator
        self.early_stopping = self._init_early_stopping()

    def __str__(self): 
        title = 'Training settings  :' + '\n' + '\n'
        
        net               = 'Net................................:  ' + model_config['net'] + '\n' 
        optimizer         = 'Optimizer..........................:  ' + model_config['optimizer'] + '\n'
        scheduler         = 'Learning Rate Scheduler............:  ' + model_config['scheduler'] + '\n'
        nb_epochs         = 'Number of epochs...................:  ' + str(train_config['nb_epochs']) + '\n'
        steps_per_epoch   = 'Number of steps_per_epoch..........:  ' + str(train_config['steps_per_epoch']) + '\n'
        return (80*'_' + '\n' + title + net + optimizer + scheduler + nb_epochs + steps_per_epoch + '\n' + 80*'_')
    
    def _init_early_stopping(self):
        if train_config['use_early_stopping'] :
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=train_config['delta'], 
                                                            patience=train_config['patience'], verbose=train_config['early_stopping_verbose'], 
                                                            mode='auto', baseline=None, restore_best_weights=False)
        return early_stopping
    

    def train(self, model_checkpoint):
        print('Starting training...')
        if train_config['use_early_stopping'] :
            callbacks_list = [model_checkpoint, self.model.scheduler, self.early_stopping]
        else :
            callbacks_list = [model_checkpoint, self.model.scheduler]

        self.model.net.fit_generator(self.train_generator,steps_per_epoch=train_config['steps_per_epoch'], callbacks=callbacks_list,
                                epochs=train_config['nb_epochs'], verbose=train_config['train_verbose']) #  callbacks=[callbacks_list[0]],
    
    def predict(self):
        print('Starting prediction...')
        return self.model.net.predict_generator(self.test_generator,train_config['nb_samples_to_predict'],verbose=1)
