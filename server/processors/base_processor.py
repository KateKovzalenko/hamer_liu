from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """Abstract base class for all processors."""

    @abstractmethod
    def process_image_file(self, file):
        """Process an uploaded image file-like object (from Flask)."""
        pass

    @abstractmethod
    def process_video_file(self, file):
        """Process an uploaded video file-like object (from Flask)."""
        pass
