import importlib
import sys
import types
from pathlib import Path

# Robust aliasing: import the target modules from the installed packages
# and insert them into sys.modules under the `keras.src.*` names. Also
# ensure parent packages exist in sys.modules so import machinery treats
# the aliased names as valid package paths.
aliases = {
    "keras.src.models.functional": "keras.models.functional",
    "keras.src.models": "keras.models",
    "keras.src.layers": "keras.layers",
}

for alias, target in aliases.items():
    try:
        real_mod = importlib.import_module(target)
        # Ensure parent packages exist as module objects
        parts = alias.split('.')
        for i in range(1, len(parts)):
            parent = '.'.join(parts[:i])
            if parent not in sys.modules:
                sys.modules[parent] = types.ModuleType(parent)
        # Register the real module under the alias
        sys.modules[alias] = real_mod
    except Exception:
        # best-effort aliasing; continue if import fails
        pass

# Run the existing evaluate script in-process so it uses these aliases.
import runpy
runpy.run_path(str(Path('src') / 'evaluate.py'), run_name='__main__')
