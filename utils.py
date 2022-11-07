import os
import pickle, time
import torch
from DataParsing.DataParsing import Match, GameState

root = os.path.dirname(os.path.abspath(__file__))

def load_data(filename):
    st = time.time()
    sample = os.path.join(root, filename)
    with (open(sample,'rb')) as openfile:  # read the parsed data
        print('reading')
        while True:
            try:
                data=pickle.load(openfile)
            except EOFError:
                break
        print('loaded')
    et = time.time()
    print('Reading time:\t',et-st) # loading time
    return data


def label_converter(label):
    onehot_label = torch.zeros(3, dtype=torch.float32)
    if label == 1:
        onehot_label[0] = 1
    elif label == 0:
        onehot_label[1] = 1
    elif label == -1:
        onehot_label[2] = 1
    onehot_label = torch.unsqueeze(onehot_label, 0)
    return onehot_label


def meta_data_converter(meta_data):
    meta_data_vector = torch.zeros(6)
    meta_data_vector[0] = meta_data["period"]
    print(type(meta_data["timestamp"]))
    meta_data_vector[2] = meta_data["actor_team"]
    meta_data_vector[3] = meta_data["actor_team_doing"]
    meta_data_vector[4] = meta_data["current_score"][0]
    meta_data_vector[5] = meta_data["current_score"][1]
    # print(meta_data_vector)
    return meta_data_vector