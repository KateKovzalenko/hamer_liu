from enum import Enum
from .mediapipe_processor import MediapipeProcessor
from .hamer_processor import HamerProcessor

class ProcessorType(Enum):
    MEDIAPIPE = "mediapipe"
    HAMER = "hamer"


def create_processor(processor_type: ProcessorType):
    """Create a processor instance based on ProcessorType enum."""
    if processor_type == ProcessorType.MEDIAPIPE:
        return MediapipeProcessor()
    elif processor_type == ProcessorType.HAMER:
        return HamerProcessor()
    else:
        raise ValueError(f"Unsupported processor type: {processor_type}")