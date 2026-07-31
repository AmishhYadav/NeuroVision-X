# Import so the `@register_model` decorators in baseline.py run on package
# import, populating the registry before `build_model` is ever called.
from neurovision.models import baseline  # noqa: F401
