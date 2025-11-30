import torch

SEED = 0
LEARN_RATE = 1e-3
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 64
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
