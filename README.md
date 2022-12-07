# joint-optimization
Source code for our journal paper [Joint Optimization of Linear and Nonlinear Models for Sequential Regression](https://www.sciencedirect.com/science/article/pii/S1051200422004195), accepted by Digital Signal Processing, Elsevier, 2022.

### Sample Usage
(Additional comments and instructions can be found in the corresponding model code)

For SARIMAX_SGD:

```python

"""
Model Run
----------
X: (numpy array) of size (sample_length x feature_size)
y: (numpy array) of size (sample_length x 1)
mean: (int) used for mean-variance normalization of y, 
            note that for convergens sampling of X columns are highly encouraged
std: (int) used for mean-variance normalization of y,
           note that for convergens sampling of X columns are highly encouraged
order: order of the model
seas_order: seasonal order of the model
optimizer: optimizer to be used in finding weights
num_epochs: int, kept at 1 for online learning
batch_size: int, kept at 1 for online learning
"""

from models import SARIMAX_SGD

model = SARIMAX_SGD(y, X, mean, std, "n", order=(0, 1, 3), seas_order=(3, 0, 3, 7))
optimizer = optim.SGD(model.parameters(), lr=0.01)

my_preds = model.train_(num_epochs=1, optimizer=optimizer, batch_size=1)
```
