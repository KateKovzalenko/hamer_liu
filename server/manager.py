from flask import Flask, request, jsonify, render_template
from server.processors.processor_factory import ProcessorType, create_processor
import cv2
import numpy as np

class Manager:
    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        # --- Add Flask configuration ---
        self.app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB uploads
        self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
        self.app.config['JSON_AS_ASCII'] = False
        # Optional: if you stream large files or responses
        self.app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        self.processor_mp = create_processor(ProcessorType.MEDIAPIPE)
        self.hamer_processor = create_processor(ProcessorType.HAMER)
        self._register_routes()

    def _register_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/upload/image', methods=['POST'])
        def upload_image():
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "No file uploaded"}), 400
            result = self.hamer_processor.process_image_file(file)
            return jsonify(result)
        """def upload_image():
            file = request.files.get('file')
            if not file:
                return jsonify({"error": "No file uploaded"}), 400
            return jsonify(self.processor_mp.process_image_file(file))"""
        
        @self.app.route('/upload/video', methods=['POST'])
        def upload_video():
            file = request.files.get('file')
            if not file:
                return jsonify({"error": "No file uploaded"}), 400
            return jsonify(self.hamer_processor.process_video_file(file))
            
    def run(self):
        self.app.run(host=self.host, port=self.port)