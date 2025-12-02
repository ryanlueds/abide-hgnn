from training_pyg.training import train_pyg

train_pyg(cv=True, ablation=False)
train_pyg(cv=False, ablation=False)
train_pyg(cv=True, ablation=True)
train_pyg(cv=False, ablation=True)