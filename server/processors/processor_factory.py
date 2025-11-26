from enum import Enum
from .hamer_processor import HamerProcessor

class ProcessorType(Enum):
    HAMER = "hamer"


def create_processor(processor_type: ProcessorType):
    """Create a processor instance based on ProcessorType enum."""
    if processor_type == ProcessorType.HAMER:
        return HamerProcessor()
    else:
        raise ValueError(f"Unsupported processor type: {processor_type}")