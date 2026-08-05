'''Shared names and defaults that do not depend on model implementations.'''

SUPPORTED_MODEL_NAMES = (
    'lightgbm', 'catboost', 'tabm', 'realmlp', 'xrfm',
)
DEFAULT_MODEL_NAMES = ('lightgbm', 'catboost')
ITERATION_MODEL_NAMES: tuple[str, ...] = ()
EPOCH_MODEL_NAMES = SUPPORTED_MODEL_NAMES
