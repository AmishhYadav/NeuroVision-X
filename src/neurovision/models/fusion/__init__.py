# Import so the `@register_fusion` decorators in adaptive_fusion.py run on
# package import, populating the registry before `build_fusion` is ever
# called. Mirrors neurovision/models/__init__.py.
from neurovision.models.fusion import adaptive_fusion  # noqa: F401
