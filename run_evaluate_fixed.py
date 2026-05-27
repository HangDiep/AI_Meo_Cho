#!/usr/bin/env python3
"""
Wrapper to fix Keras model deserialization for models saved with keras.src.models.functional references.
Patches TensorFlow/Keras deserialization before running src/evaluate.py
"""
import sys
import os
import runpy
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.src import models as tf_keras_src_models

# Patch the deserialization function to handle keras.src.models.functional
original_get_registered_object = None

def patched_get_registered_object(identifier, custom_objects, module_objects, printable_module_name):
    """
    Patched version that handles keras.src.models.functional references
    """
    # If it's asking for keras.src.models.functional, return tf.keras.models.Functional
    if identifier == 'keras.src.models.functional.Functional':
        try:
            # Try to get Functional from keras.src.models.functional directly
            return tf_keras_src_models.functional.Functional
        except (AttributeError, ImportError):
            # Fallback: create wrapper that returns the actual Functional implementation
            try:
                from tensorflow.keras.models import Functional as KerasFunctional
                return KerasFunctional
            except ImportError:
                # Last resort: check if it exists in tf.keras.models internals
                if hasattr(models, 'Functional'):
                    return models.Functional
                # If still not found, let original handler deal with it
                return original_get_registered_object(identifier, custom_objects, module_objects, printable_module_name)
    
    # For other identifiers, use original function
    return original_get_registered_object(identifier, custom_objects, module_objects, printable_module_name)

# Apply patches before importing model loading code
try:
    from tensorflow.keras.saving import serialization_lib
    original_get_registered_object = serialization_lib.get_registered_object
    serialization_lib.get_registered_object = patched_get_registered_object
    print("✓ Patched TensorFlow/Keras deserialization for keras.src.* compatibility")
except Exception as e:
    print(f"⚠ Warning: Could not patch deserialization: {e}")

# Add custom object handler that maps keras.src.* to available modules
def custom_objects_loader():
    """Return a dict of custom objects for loading model with keras.src.* references"""
    custom_objs = {}
    
    # Add Functional if available
    try:
        from tensorflow.keras.src.models.functional import Functional
        custom_objs['Functional'] = Functional
    except (ImportError, AttributeError):
        try:
            # Fallback to tf.keras
            from tensorflow.python.keras.api._v2.keras.models import Functional
            custom_objs['Functional'] = Functional
        except (ImportError, AttributeError):
            pass
    
    return custom_objs

print("=" * 70)
print("Running src/evaluate.py with keras.src.models.functional compatibility fix")
print("=" * 70)

# Run evaluate.py
try:
    runpy.run_path('src/evaluate.py', run_name='__main__')
except SystemExit as e:
    sys.exit(e.code)
except Exception as e:
    print(f"\n✗ Error running evaluate: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
