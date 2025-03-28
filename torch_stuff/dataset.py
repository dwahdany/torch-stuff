from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset, Subset


class DelegatingSubset(Subset):
    """A Subset class that delegates all dataset attributes to the underlying dataset.

    This class inherits from torch.utils.data.Subset and ensures that all dataset
    attributes (like targets, classes, etc.) are properly delegated to the underlying
    dataset. This is particularly useful when working with datasets that have custom
    attributes that need to be preserved when creating subsets.
    """

    def __init__(self, dataset: Dataset, indices: List[int]):
        """Initialize the DelegatingSubset.

        Args:
            dataset: The dataset to create a subset from
            indices: The indices to include in the subset
        """
        super().__init__(dataset, indices)

    def __getattr__(self, name: str) -> Any:
        """Delegate any attribute access to the underlying dataset.

        This ensures that any attributes not directly defined in this class
        (like targets, classes, etc.) are accessed from the underlying dataset.

        Args:
            name: The name of the attribute to access

        Returns:
            The attribute value from the underlying dataset

        Raises:
            AttributeError: If the attribute doesn't exist in the underlying dataset
        """
        return getattr(self.dataset, name)

    @property
    def targets(self) -> torch.Tensor:
        """Get the targets for the subset indices.

        Returns:
            The targets for the selected indices
        """
        return self.dataset.targets[self.indices]

    @property
    def classes(self) -> List[str]:
        """Get the classes from the underlying dataset.

        Returns:
            The list of classes
        """
        return self.dataset.classes

    def __len__(self) -> int:
        """Get the length of the subset.

        Returns:
            The number of items in the subset
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the subset.

        Args:
            idx: The index of the item to get

        Returns:
            A tuple of (data, target) for the selected index
        """
        return self.dataset[self.indices[idx]]
