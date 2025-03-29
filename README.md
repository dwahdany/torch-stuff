# torch-stuff

A collection of useful PyTorch utilities and dataset helpers.

## Installation

```bash
pip install torch-stuff
```

## Features

### DelegatingSubset

A PyTorch dataset subset class that properly delegates all dataset attributes to the underlying dataset. This is particularly useful when working with datasets that have custom attributes that need to be preserved when creating subsets.

```python
from torch_stuff.dataset import DelegatingSubset
from torch.utils.data import Dataset

# Create your dataset
class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(100, 3, 32, 32)
        self.targets = torch.randint(0, 10, (100,))
        self.classes = ['class1', 'class2', 'class3']  # Custom attribute
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Create a subset that preserves all attributes
dataset = MyDataset()
subset = DelegatingSubset(dataset, indices=[0, 1, 2, 3])

# Access custom attributes from the subset
print(subset.classes)  # ['class1', 'class2', 'class3']
```
