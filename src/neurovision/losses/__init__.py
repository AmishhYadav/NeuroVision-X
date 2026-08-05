# Import so the `@register_loss` decorator in segmentation.py / multitask.py
# runs on package import, populating the registry before `build_loss` is ever
# called.
from neurovision.losses import (
    multitask,  # noqa: F401
    segmentation,  # noqa: F401
)
