from typing import List, Tuple

import pytest
import torch
from torch.utils.data import Dataset

from torch_stuff.dataset import DelegatingSubset


class MockDataset(Dataset):
    """A mock dataset for testing purposes."""

    def __init__(self, data: torch.Tensor, targets: torch.Tensor, classes: List[str]):
        self.data = data
        self.targets = targets
        self.classes = classes
        self.custom_attribute = "custom_value"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    data = torch.randn(10, 3, 32, 32)  # 10 images of size 32x32 with 3 channels
    targets = torch.randint(0, 3, (10,))  # 10 targets with 3 classes
    classes = ["class1", "class2", "class3"]
    return MockDataset(data, targets, classes)


def test_delegating_subset_initialization(mock_dataset):
    """Test that DelegatingSubset initializes correctly."""
    indices = [0, 1, 2]
    subset = DelegatingSubset(mock_dataset, indices)
    assert len(subset) == 3
    assert subset.indices == indices
    assert subset.dataset == mock_dataset


def test_delegating_subset_targets(mock_dataset):
    """Test that targets are correctly indexed."""
    indices = [0, 1, 2]
    subset = DelegatingSubset(mock_dataset, indices)
    expected_targets = mock_dataset.targets[indices]
    assert torch.equal(subset.targets, expected_targets)


def test_delegating_subset_classes(mock_dataset):
    """Test that classes are correctly delegated."""
    subset = DelegatingSubset(mock_dataset, [0, 1])
    assert subset.classes == mock_dataset.classes


def test_delegating_subset_getitem(mock_dataset):
    """Test that __getitem__ returns correct data and targets."""
    indices = [0, 1]
    subset = DelegatingSubset(mock_dataset, indices)

    # Test first item
    data, target = subset[0]
    expected_data, expected_target = mock_dataset[indices[0]]
    assert torch.equal(data, expected_data)
    assert torch.equal(target, expected_target)

    # Test second item
    data, target = subset[1]
    expected_data, expected_target = mock_dataset[indices[1]]
    assert torch.equal(data, expected_data)
    assert torch.equal(target, expected_target)


def test_delegating_subset_custom_attribute(mock_dataset):
    """Test that custom attributes are correctly delegated."""
    subset = DelegatingSubset(mock_dataset, [0])
    assert subset.custom_attribute == mock_dataset.custom_attribute


def test_delegating_subset_nonexistent_attribute(mock_dataset):
    """Test that accessing nonexistent attributes raises AttributeError."""
    subset = DelegatingSubset(mock_dataset, [0])
    with pytest.raises(AttributeError):
        _ = subset.nonexistent_attribute


def test_delegating_subset_empty_indices(mock_dataset):
    """Test that DelegatingSubset works with empty indices."""
    subset = DelegatingSubset(mock_dataset, [])
    assert len(subset) == 0
    assert subset.classes == mock_dataset.classes  # Should still have access to classes
