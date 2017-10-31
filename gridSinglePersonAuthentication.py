import os
import re
import glob
import logging
import numpy as np
from keras.preprocessing.image import * 

from gridBaseDataset import GRIDBaseDataset

class GRIDSinglePersonAuthentication(GRIDBaseDataset):
    def __init__(self, auth_person=23, pos_n=25, *args, **kwargs):
        GRIDBaseDataset.__init__(self, *args, **kwargs)
        self.num_classes = 2
        self.auth_person = auth_person
        neg_people = np.arange(1,20)
        logging.info( 'getting positive and negative lip paths...')
        logging.info( 'training positive sample number is : {}'.format(pos_n) ) 
        self.neg_lip_paths = self.getLipPaths(neg_people) 
        self.pos_lip_paths = self.getLipPaths( (auth_person, ) ) 
        #fix seed to keep the same splition of train dataset
        np.random.seed(100)
        np.random.shuffle(self.pos_lip_paths) 
        np.random.shuffle(self.neg_lip_paths) 
        np.random.seed()

        self.augmenter = ImageDataGenerator(
                rotation_range=5, 
                width_shift_range = 0.1,
                height_shift_range=0.1,
                # shear_range=0.05,
                zoom_range=0.1,
                horizontal_flip=True
                # rescale=1./255
                ) 

        split = int(len(self.neg_lip_paths)*0.9)
        other_people = np.arange(22,34) 
        unseen_neg_people = [] 
        for person in other_people:
            if person != auth_person:
                unseen_neg_people.append(person) 

        self.pos_train_paths = self.pos_lip_paths[0:pos_n] 
        self.neg_train_paths = self.neg_lip_paths[0:split] 

        self.pos_val_paths = self.pos_lip_paths[pos_n:pos_n+25] 
        self.neg_val_paths = self.neg_lip_paths[split:] 

        self.pos_test_paths = self.pos_lip_paths[pos_n+25:] 
        self.neg_test_paths = self.getLipPaths(unseen_neg_people ) 

    def next_batch(self, batch_size, phase, gen_words=False, shuffle=False):
        if phase == 'train':
            pos_paths = self.pos_train_paths
            neg_paths = self.neg_train_paths 
        elif phase == 'val':
            pos_paths = self.pos_val_paths
            neg_paths = self.neg_val_paths
        elif phase == 'test':
            pos_paths = self.pos_test_paths
            neg_paths = self.neg_test_paths
        else:
            raise ValueError( 'phase must be one of {train, val, test}') 
        pos_n  = len(pos_paths) 
        nb_iterate = int(np.ceil(float(len(pos_paths))/(batch_size // 2) ))
        neg_path_idx = 0
        neg_path_nums = len(neg_paths) 
        np.random.seed(100)
        while True:
            if self.shuffle:
                np.random.shuffle(pos_paths)
                np.random.shuffle(neg_paths) 
            start_pos = 0
            for itr in range(nb_iterate):
                current_batch_size = batch_size
                if pos_n < start_pos + batch_size//2:
                    current_batch_size = (pos_n - start_pos) * 2
                batch_paths = []
                batch_paths.extend(pos_paths[start_pos: start_pos+current_batch_size//2] )
                if neg_path_idx + current_batch_size // 2 > neg_path_nums:
                    neg_path_idx = 0
                    if shuffle:
                        np.random.shuffle(neg_paths) 
                batch_paths.extend(neg_paths[neg_path_idx: neg_path_idx+current_batch_size//2] )
                start_pos += current_batch_size // 2
                neg_path_idx += current_batch_size // 2 
                if phase == 'train' or phase == 'val':
                    augmenter = self.augmenter
                else:
                    augmenter = None
                yield self.gen_batch(0, current_batch_size, batch_paths, gen_words=gen_words, auth_person=self.auth_person, scale=1./255, augmenter=augmenter)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)  
    gridPersonDatabase = GRIDSinglePersonAuthentication( debug = True)
    batch_size=4
    print ('gen a train batch.........')
    next(gridPersonDatabase.next_batch(batch_size, phase='train', shuffle = True))
    [1 for value in gridPersonDatabase.next_batch(batch_size, phase='train', shuffle = True)]
    print ('gen a val batch.........')
    next(gridPersonDatabase.next_batch(batch_size, phase='val', shuffle = False))
