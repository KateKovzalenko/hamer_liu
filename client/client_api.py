import requests
import os

class ClientAPI:
    """
    Handles network communication with the Hand Tracking API.
    """
    def __init__(self, base_url, image_timeout=(10, 600), video_timeout=(10, 3600)):
        """
        Initializes the client.
        
        Parameters:
        base_url (str): The base URL of the API (e.g., "http://localhost:8080").
        image_timeout (tuple): (connect, read) timeout for image uploads.
        video_timeout (tuple): (connect, read) timeout for video uploads.
        """
        self.base_url = base_url
        self.image_url = f"{base_url}/upload/image"
        self.video_url = f"{base_url}/upload/video"
        self.image_timeout = image_timeout
        self.video_timeout = video_timeout

    def upload_image(self, image_path):
        """
        Sends an image file to the /upload/image endpoint.
        
        Parameters:
        image_path (str): Path to the image file.
        
        Returns:
        dict: The JSON response from the server.
        
        Raises:
        requests.exceptions.RequestException: On connection error or HTTP error.
        """
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(
                self.image_url, 
                files=files, 
                timeout=self.image_timeout
            )
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()

    def upload_video(self, video_path):
        """
        Sends a video file to the /upload/video endpoint.
        
        Parameters:
        video_path (str): Path to the video file.
        
        Returns:
        dict: The JSON response from the server.
        
        Raises:
        requests.exceptions.RequestException: On connection error or HTTP error.
        """
        with open(video_path, 'rb') as f:
            files = {'file': (os.path.basename(video_path), f, 'video/mp4')}
            response = requests.post(
                self.video_url, 
                files=files, 
                timeout=self.video_timeout
            )
        response.raise_for_status()
        return response.json()