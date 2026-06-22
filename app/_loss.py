"""Stub module for unpickling models trained with custom loss functions.

This module provides placeholder loss function classes/objects that allow
pickled models to deserialize without errors, even if the original loss
implementation is not available.
"""

# Placeholder for any custom loss classes that were used during training
class CustomLoss:
    """Stub for custom loss functions."""
    def __call__(self, *args, **kwargs):
        return 0

# Create module-level instances that pickle might be looking for
loss = CustomLoss()
