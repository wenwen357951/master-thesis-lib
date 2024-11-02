import tensorflow as tf

from .ich_420_dataset import ICH420Dataset
from .config import Config

# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(physical_devices), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

__all__ = [
    'ICH420Dataset',
    'Config'
]
