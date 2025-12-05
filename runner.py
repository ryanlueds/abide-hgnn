from training_pyg.training import train_pyg
from training_mlp.training import train_mlp
from training_dhg.training import train_dhg

train_mlp(cv=True)
train_mlp(cv=False)

train_pyg(cv=True, ablation=False)
train_pyg(cv=False, ablation=False)
train_pyg(cv=True, ablation=True)
train_pyg(cv=False, ablation=True)

train_dhg(cv=True, ablation=False)
train_dhg(cv=False, ablation=False)
train_dhg(cv=True, ablation=True)
train_dhg(cv=False, ablation=True)