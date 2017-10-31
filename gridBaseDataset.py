import os
import numpy as np
import h5py
import keras.backend as K
import glob
from PIL import Image as pil_image
from keras.preprocessing.image import *

class GRIDBaseDataset(object):
    def __init__(self, target_size=[50,100], shuffle=True, re_generate=False, data_dir='./data/GRID', debug=False):
        self.data_root = data_dir
        self.lip_dir = data_dir+'/lip'
        self.label_dir = data_dir+'/alignments'
        self.re_generate = re_generate
        self.timespecs = 75
        self.ctc_blank = 27
        self.target_size = target_size
        self.max_label_length = 50
        self.shuffle=shuffle

        self.input_dim = (self.timespecs, target_size[0], target_size[1], 3)
        self.output_dim = (self.max_label_length)

        self.debug = debug

        self.train_paths = None
        self.test_seen_paths = None
        self.test_unseen_paths = None

    def get_speaker_idx_of_path(self, one_path):
        match = re.match(r'.*\/s(\d+)\/.*',one_path)
        return int(match.group(1)) - 1

    def gen_batch(self, begin, batch_size, paths,gen_words, auth_person=None, scale=1., augmenter=None):

        data = np.zeros([batch_size, self.timespecs, self.target_size[0], self.target_size[1], 3])
        label = np.zeros([batch_size, self.max_label_length])
        input_length = np.zeros([batch_size, 1])
        label_length = np.zeros([batch_size, 1])
        source_strs = []

        if auth_person:
            y_person = np.zeros([batch_size, self.timespecs, 2] ) 
        else:
            y_person = np.zeros([batch_size, self.timespecs, 34] ) 

        for i in range(batch_size):
            pos = begin+i
            lip_d, lip_l, lip_label_len, source_str = self.readLipSequences(paths[pos], gen_words=gen_words)
            data[i] = lip_d * scale
            label[i] = lip_l
            input_length[i] = self.timespecs - 2
            label_length[i] = lip_label_len
            source_strs.append(source_str)
            if auth_person:
                if self.get_speaker_idx_of_path(paths[pos]) == auth_person-1:
                    y_person[i,:,1] = 1
                else:
                    y_person[i,:,0] = 1 
            else:
                y_person[i, :, self.get_speaker_idx_of_path(paths[pos])] = 1

        inputs = {'inputs': data,
                'labels': label,
                'input_length': input_length,
                'label_length': label_length,
                'source_str': source_strs
                }
        if auth_person:
            y_person_name = 'y_person_2'
        else:
            y_person_name = 'y_person_34'
        outputs = {'ctc':np.zeros([batch_size]), y_person_name: y_person}
        if augmenter:
            data = inputs[ 'inputs'] 
            bs, timespec = data.shape[0:2] 
            for b in range(bs):
                seed = np.random.randint(100000) 
                for t in range(timespec):
                    np.random.seed(seed)
                    data[b,t,...] = augmenter.random_transform(data[b,t,...] )
                np.random.seed() 
        return (inputs, outputs)

    """
    convert a word comprised of characters to an tuple of integer a -> 0
    b -> 1
    z -> 25

    example:
    one -> (14, 13, 4)
    """
    def convertWordToLabel(self, string):
        label = []
        for char in string:
            if char == ' ':
                label.append(26)
            else:
                label.append(ord(char) - ord('a'))
        return label

    """
    convert align file to label:

    an align file looks like:

    ```
    0 23750 sil
    23750 29500 bin
    29500 34000 blue
    34000 35500 at
    35500 41000 f
    41000 47250 two
    47250 53000 now
    53000 74500 sil
    ```
    so the word list is (bin, blue, at, f, two, now) then convert it to interger tuple.

    @return
    the returned tuple has four elements, each of which is a list. The first element of the list is the whole sentence, and the following are single words.
    labels: each element is an integer array
    labels_len: each element is the length of the label length
    source_strs: the original word
    frames: the frame begin and end

    """
    def convertAlignToLabels(self, align_file):
        with open(align_file,'r') as f:
            lines = f.readlines()
            words = []
            frames = [[1,self.timespecs]]
            sentence_label = []
            source_strs_of_words = []
            source_str = ''
            source_strs = []
            for i,line in enumerate(lines):

                #remove first and last SIL word
                if i ==0:
                    continue
                if i == len(lines)-1:
                    continue

                striped_line = line.rstrip()
                begin,end,word = striped_line.split(' ')
                if word == 'sp':
                    continue
                source_str += word
                sentence_label.extend(self.convertWordToLabel(word))
                begin_frame = int(begin) // 1000+1
                end_frame = int(end) // 1000+1
                frames.append([begin_frame, end_frame])
                if i!=len(lines)-2:
                    sentence_label.append(26)
                    source_str += ' '

                word = ' ' + word + ' '
                words.append(word)

            labels = np.zeros([len(words)+1, self.max_label_length])-1
            labels_len = np.zeros(len(words)+1)
            labels[0, 0:len(sentence_label)] = sentence_label
            labels_len[0] = len(sentence_label)
            source_strs.append(source_str)
            for i,w in enumerate(words):
                labels[i+1,0:len(w)] = self.convertWordToLabel(w)
                labels_len[i+1] = len(w)
                source_strs.append(w)

        return (labels, labels_len, source_strs, frames)

    """
    load an image and convert each pixel value to range of (-1,1)
    """
    def load_image(self, path, grayscale=False, target_size=None):
        img = pil_image.open(path)
        if grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        if target_size:
            img = img.resize((target_size[1],target_size[0]))
        img = np.asarray(img, dtype=float)
        return img


    """
    read a lip sequence to numpy array
    """
    def readLipSequences(self, lipsequence_dir, gen_words):
        total = len(sum([i[2] for i in os.walk(lipsequence_dir)],[]))

        sequence_name = lipsequence_dir.split('/')[-1]
        sequence_owner = lipsequence_dir.split('/')[-2]
        lip_sequence = np.zeros([self.timespecs, self.target_size[0], self.target_size[1], 3])
        if total < self.timespecs-1:
            label = np.zeros(self.max_label_length)-1
            label[0] = self.ctc_blank
            return (lip_sequence, label, 1, "")

        def read_images(frame_intval):
            begin_frame, end_frame = frame_intval
            for i in range(begin_frame, end_frame+1):
                img_name = '{}/{}.jpeg'.format(lipsequence_dir, i)
                lip_sequence[i-begin_frame,...] = img_to_array(load_img(img_name, target_size=self.target_size))

        label_path = self.getAlignmentDirOfPerson(sequence_owner, sequence_name)
        labels, labels_len, source_strs, frames = self.convertAlignToLabels(label_path)
        i = 0
        if gen_words and np.random.rand() < 0.3:
            i = np.random.randint(len(frames))
        read_images(frames[i])
        return (lip_sequence, labels[i], labels_len[i],source_strs[i])

    def getLipDirOfPerson(self, i):
        return "{}/lip/s{}".format(self.data_root, i)

    def getAlignmentDirOfPerson(self, i, name):
        return "{}/alignments/{}/align/{}.align".format(self.data_root, i, name)

    def getLipPaths(self, people):
        paths = []
        for i in people:
            person_lip_dir = self.getLipDirOfPerson(i)
            lip_dirs = glob.glob(person_lip_dir+'/*')
            paths.extend(lip_dirs)
        return paths
