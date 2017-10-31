import os
import sys
import numpy as np
from time import gmtime, strftime
from keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler

from model.lipnet_res3d import lipnet_res3d
from model.stastic_callbacks import StatisticCallback
from gridDatasetGenerator import GRIDDatasetGenerator 

batch_size = 25
nb_epoch = 500
name =  'multiuser_with_auth'
initial_epoch = 0
timestamp=strftime("%Y%m%d_%H%M",gmtime())
weight_savepath = './data/checkpoints_{}/lipnet_res3d_weights_{}.hdf5'.format(timestamp, name)
log_savepath='./data/logs_{}/lipnet_res3d_loss_seen_{}.csv'.format(timestamp, name)
log_savepath_unseen = './data/logs_{}/lipnet_res3d_loss_unseen_{}.csv'.format(timestamp, name)
trainval_loss_savepath = './data/logs_{}/lipnet_res3d_trainval_loss_{}.csv'.format(timestamp, name)

os.system( 'mkdir -p ./data/checkpoints_{}'.format(timestamp) ) 
os.system( 'mkdir -p ./data/logs_{}'.format(timestamp) ) 

grid = GRIDDatasetGenerator()
net,test_func = lipnet_res3d(grid.input_dim, grid.output_dim, weights=None) 
net.summary()

#callbacks
checkpointer = ModelCheckpoint(filepath=weight_savepath,save_best_only=True,save_weights_only=True)
csv = CSVLogger(trainval_loss_savepath)
nb_train_samples = grid.train_num
#generators
train_gen = grid.next_batch(batch_size, phase= 'train', gen_words=False)
val_gen_seen = grid.next_batch(batch_size, phase= 'val', test_seen=True, gen_words=False)
val_gen_unseen = grid.next_batch(batch_size, phase= 'val', test_seen=False, gen_words=False)

statisticCallback = StatisticCallback(test_func, log_savepath, val_gen_seen, grid.test_seen_num, weight_savepath)
statisticCallback_unseen = StatisticCallback(test_func, log_savepath_unseen, val_gen_unseen, grid.test_unssen_num, None)
net.fit_generator(generator=train_gen, 
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=nb_epoch,initial_epoch=initial_epoch,
                    validation_data=val_gen_seen, validation_steps=grid.test_seen_num // batch_size,
                    callbacks=[statisticCallback, statisticCallback_unseen, csv]
                    # use_multiprocessing=True, workers=4
                    )
