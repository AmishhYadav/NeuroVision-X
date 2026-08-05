# Import so the `@register_model` decorators run on package import, populating the
# registry before `build_model` is ever called: baseline.py registers "unet3d" and
# "swinunetr", neurovision.py registers "neurovision" (the full dual-encoder network).
from neurovision.models import (
    baseline,  # noqa: F401
    neurovision,  # noqa: F401
)
