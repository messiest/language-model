import random
import argparse

import torch


parser = argparse.ArgumentParser(
    description='WikiText Language Model'
)
parser.add_argument(
    '--data',
    type=str,
    default='./data/wikitext-2',
    help='location of the data corpus',
)
parser.add_argument(
    '--model',
    type=str,
    default='LSTM',
    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)',
)
parser.add_argument(
    '--emsize',
    type=int,
    default=200,
    help='size of word embeddings',
)
parser.add_argument(
    '--nhid',
    type=int,
    default=200,
    help='number of hidden units per layer',
)
parser.add_argument(
    '--nlayers',
    type=int,
    default=2,
    help='number of layers',
)
parser.add_argument(
    '--lr',
    type=float,
    default=20,
    help='initial learning rate',
)
parser.add_argument(
    '--clip',
    type=float,
    default=0.25,
    help='gradient clipping',
)
parser.add_argument(
    '--epochs',
    type=int,
    default=40,
    help='upper epoch limit',
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=20,
    metavar='N',
    help='batch size',
)
parser.add_argument(
    '--bptt',
    type=int,
    default=35,
    help='sequence length',
)
parser.add_argument(
    '--dropout',
    type=float,
    default=0.2,
    help='dropout applied to layers (0 = no dropout)',
)
parser.add_argument(
    '--tied',
    action='store_true',
    help='tie the word embedding and softmax weights'
)
parser.add_argument(
    '--seed',
    type=int,
    default=random.randint(0, 10000),  # 1111
    help='random seed'
)
parser.add_argument(
    '--cuda',
    action='store_true',
    default=torch.cuda.is_available(),
    help='use CUDA'
)
parser.add_argument(
    '--log-interval',
    type=int,
    default=100,
    metavar='N',
    help='report interval'
)
parser.add_argument(
    '--save',
    type=str,
    default='model.pt',
    help='path to save the final model'
)

def make():
    return parser.parse_args()
