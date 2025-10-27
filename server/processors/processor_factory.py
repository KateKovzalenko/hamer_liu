from enum import Enum
from .mediapipe_processor import MediapipeProcessor
# from .hammer_processor import HammerProcessor  # optional later


class ProcessorType(Enum):
    MEDIAPIPE = "mediapipe"
    # HAMMER = "hammer"  # to be implemented later


def create_processor(processor_type: ProcessorType):
    """Create a processor instance based on ProcessorType enum."""
    if processor_type == ProcessorType.MEDIAPIPE:
        return MediapipeProcessor()
    # elif processor_type == ProcessorType.HAMMER:
    #     return HammerProcessor()
    else:
        raise ValueError(f"Unsupported processor type: {processor_type}")