from training_pyg.training import train_pyg
from training_mlp.training import train_mlp

# train_pyg(cv=True, ablation=False)
# train_pyg(cv=False, ablation=False)
# train_pyg(cv=True, ablation=True)
# train_pyg(cv=False, ablation=True)

train_mlp(cv=True)
train_mlp(cv=False)