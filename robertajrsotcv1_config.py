import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

BATCH_SIZE = 4
EPOCHS = 4
LEARNING_RATE = 1e-05
MAX_LEN = 400

params = {'batch_size': BATCH_SIZE,
          'shuffle': True
          }

device = 'cuda' if cuda.is_available() else 'cpu'
tokenizer = RobertaTokenizer.from_pretrained('roberta-base',
                                             truncation=True)
loss_function = torch.nn.MSELoss()



