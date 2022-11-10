import os
import pickle, time
import torch
from DataParsing.DataParsing import Match, GameState
from itertools import chain
import random

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


def label_to_onehot(label):
    onehot_label = torch.zeros(3, dtype=torch.float32)
    if label == 1:
        onehot_label[0] = 1
    elif label == 0:
        onehot_label[1] = 1
    elif label == -1:
        onehot_label[2] = 1
    onehot_label = torch.unsqueeze(onehot_label, 0)
    return onehot_label


def label_to_index(label):
    if label == 1:
        return 0
    elif label == 0:
        return 1
    elif label == -1:
        return 2


def meta_data_to_vector(meta_data):
    meta_data_vector = torch.zeros(6, dtype=torch.float32)
    meta_data_vector[0] = meta_data["period"]
    timestamp = meta_data["timestamp"].split(":")
    timestamp = [float(x) for x in timestamp]
    meta_data_vector[1] = (timestamp[0]*60*60 + timestamp[1]*60 + timestamp[2]) / (30*60)
    meta_data_vector[2] = meta_data["actor_team"]
    meta_data_vector[3] = meta_data["actor_team_doing"]
    meta_data_vector[4] = meta_data["current_score"][0]
    meta_data_vector[5] = meta_data["current_score"][1]
    return meta_data_vector


def flatten_data(data):
    flat_data = []
    for match in data:
        flat_data.append(match.gamestates)
    flat_data = list(chain.from_iterable(flat_data))
    return flat_data


def get_location_data(data, match_index):
    location_data = []
    match = data[match_index]
    for timestamp in match.gamestates:
        try:
            location_tensor = timestamp.graph.x.T
        except:
            continue
        location_tensor, team_tensor = location_tensor[:2], location_tensor[2]
        home = (location_tensor.T)[team_tensor == 1].T
        away = (location_tensor.T)[team_tensor == -1].T
        location_data.append((home, away))
    return location_data
    