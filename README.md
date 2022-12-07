# joint-optimization
Source code for our journal paper [Joint Optimization of Linear and Nonlinear Models for Sequential Regression](https://www.sciencedirect.com/science/article/pii/S1051200422004195), accepted by Digital Signal Processing, Elsevier, 2022.

### Sample Usage
(Additional comments and instructions can be found in the corresponding model code. Additionally, theoretical explanation can be found in the given paper)

For SARIMAX_SGD:

```python

"""
Model Run
----------
X: (numpy array) of size (sample_length x feature_size)
y: (numpy array) of size (sample_length x 1)
mean: (float) used for mean-variance normalization of y, 
              note that for convergence sampling of X columns are highly encouraged
              should be gathered from training data only
std: (float) used for mean-variance normalization of y,
             note that for convergence sampling of X columns are highly encouraged
             should be gathered from training data only
order: order of the model
seas_order: seasonal order of the model
optimizer: optimizer to be used in finding weights
num_epochs: int, kept at 1 for online learning
batch_size: int, kept at 1 for online learning
"""

from sarimax import SARIMAX_SGD

model = SARIMAX_SGD(y, X, mean, std, "n", order=(0, 1, 3), seas_order=(3, 0, 3, 7))
optimizer = optim.SGD(model.parameters(), lr=0.01)

my_preds = model.train_(num_epochs=1, optimizer=optimizer, batch_size=1)
```

For SoftGBM:

```python

"""
Model Run
----------
X: (numpy array) of size (sample_length x feature_size)
y: (numpy array) of size (sample_length x 1)
mean: (float) used for mean-variance normalization of y, 
              note that for convergence sampling of X columns are highly encouraged
              should be gathered from training data only
std: (float) used for mean-variance normalization of y,
             note that for convergence sampling of X columns are highly encouraged
             should be gathered from training data only
optimizer: optimizer to be used in optimizing weights
num_epochs: int, kept at 1 for online learning
batch_size: int, kept at 1 for online learning
"""

from soft import SoftGBM
from torch.utils.data import DataLoader

model_dataset = []

for i in range(len(y)):
    model_dataset.append(tuple([torch.from_numpy(X[i]),torch.from_numpy(np.array(y[i]))]))
    
model_dataloader = DataLoader(tuple(model_dataset), batch_size=1, shuffle=False)

model_soft = SoftGBM(num_trees=20, tree_depth=2, input_dim=56, shrinkage_rate=0.15)
optimizer = optim.SGD(model_soft.parameters(), lr=0.005)
my_preds = model_soft.train_(model_dataloader, optimizer, num_epochs=1, mean=mean, std=std)
```

For Joint Model:

```python

"""
Model Run
----------
X: (numpy array) of size (sample_length x feature_size)
y: (numpy array) of size (sample_length x 1)
mean: (float) used for mean-variance normalization of y, 
              note that for convergence sampling of X columns are highly encouraged
              should be gathered from training data only
std: (float) used for mean-variance normalization of y,
             note that for convergence sampling of X columns are highly encouraged
             should be gathered from training data only
optimizer: optimizer to be used in optimizing weights
order: order of the sarimax model
seas_order: seasonal order of the sarimax model
num_epochs: int, kept at 1 for online learning
batch_size: int, kept at 1 for online learning
"""

from torch.utils.data import DataLoader
from joint_model import JointModel

model_dataset = []

for i in range(1,len(y)):
    model_dataset.append(tuple([torch.from_numpy(X[i]),torch.from_numpy(np.array(y[i]))]))

model_dataloader = DataLoader(tuple(model_dataset), batch_size=1, shuffle=False)

model = JointModel(num_trees=20, tree_depth=2, input_dim=56, shrinkage_rate=0.15, 
                   mean=mean, std=std, endog=y, exog=X, trend="n", order=(0, 1, 3), seas_order=(3, 0, 3, 7))

opt1 = optim.SGD(model.Linear.parameters(), lr=0.001)
opt2 = optim.SGD(model.trees.parameters(), lr=0.005)
sarimax_preds, soft_preds = model.train_(train_dataloader, num_epochs=1, batch_size=1, first_endog=first_endog, optimizer1=opt1, optimizer2=opt2)
joint_preds = sarimax_preds + soft_preds
```
