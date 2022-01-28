

import torch

def load_data():
    import tensorflow_datasets as tfds

    data = tfds.load('glue/sst2' ,split="train", shuffle_files=False)
    
    X = [str(e["sentence"].numpy()) for e in data]
    y = [int(e["label"]) for e in data]

    data = tfds.load('glue/sst2' ,split="validation", shuffle_files=False)
    X_test = [str(e["sentence"].numpy()) for e in data]
    y_test = [int(e["label"]) for e in data]
    
    return X,X_test,torch.LongTensor(y),torch.LongTensor(y_test)