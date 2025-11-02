import torch

SEED = 0
LEARN_RATE = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 64
EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
