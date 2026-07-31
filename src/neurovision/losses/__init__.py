# Import so the `@register_loss` decorator in segmentation.py runs on package
# import, populating the registry before `build_loss` is ever called.
from neurovision.losses import segmentation  # noqa: F401
