import os
import numpy as np
import h5py
import keras.backend as K
import glob
from PIL import Image as pil_image
from keras.preprocessing.image import * 

from gridBaseDataset import GRIDBaseDataset

class GRIDDatasetGenerator(GRIDBaseDataset):
    def __init__(self, test_people=(1,2,20), *args, **kwargs):
        GRIDBaseDataset.__init__(self, *args, **kwargs)
        self.test_people = test_people


        self.train_people = []
        for p in range(1,35):
            if p not in test_people:
                self.train_people.append(p)
        #train the first 20 person apart from the test person
        self.train_people = self.train_people[0:17]

        print ("train_people:{} test_people: {}".format(self.train_people, self.test_people))
        train_lip_paths = self.getLipPaths(self.train_people)

        #fix seed to keep the same splition of train dataset
        np.random.seed(10000)
        np.random.shuffle(train_lip_paths)
        np.random.seed()
        train_n = len(train_lip_paths)
        split = 0.9
        train_num = int(train_n *split)
        test_unseen_paths = self.getLipPaths(self.test_people)

        self.train_paths=train_lip_paths[0:train_num]
        self.test_seen_paths=train_lip_paths[train_num:]
        self.test_unseen_paths=test_unseen_paths

        self.train_num = train_num
        self.test_seen_num = len(self.test_seen_paths)
        self.test_unssen_num = len(self.test_unseen_paths)

        self.augmenter = ImageDataGenerator(
                rotation_range=5, 
                width_shift_range = 0.1,
                height_shift_range=0.1,
                # shear_range=0.05,
                zoom_range=0.1,
                horizontal_flip=True
                ) 

    def next_batch(self, batch_size, phase, test_seen=False,gen_words=False, shuffle=True):
        if phase == 'train':
            paths = self.train_paths
        elif phase == 'val':
            if test_seen:
                paths = self.test_seen_paths
            else:
                paths = self.test_unseen_paths
	nb_iterate = len(paths) // batch_size
        if phase == 'train' or phase == 'val':
            augmenter = self.augmenter
        else:
            augmenter = None
        while True:
            if self.shuffle:
                np.random.shuffle(paths)
            for itr in range(nb_iterate):
                start_pos = itr*batch_size
                yield self.gen_batch(start_pos, batch_size, self.train_paths, gen_words=gen_words, scale= 1./255, augmenter=augmenter)

if __name__ == '__main__':
    grid = GRIDDatasetGenerator(debug=True)
    batch_size=30
    print ('gen a train batch.........')
    import time
    t1 = time.time()
    for x in grid.next_batch(batch_size,phase= 'train',gen_words=False):
        t = time.time()
        print (t-t1) 
        t1 = t
    # x = next(grid.next_train_batch(batch_size, gen_words=False))
    # print x
    # print ('gen a val batch.........')
    # x_val = next(grid.next_val_batch(batch_size))
    # print x_val

