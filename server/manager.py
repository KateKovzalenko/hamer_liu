from flask import Flask, request, jsonify, render_template
from server.processors.processor_factory import ProcessorType, create_processor

class Manager:
    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.processor_mp = create_processor(ProcessorType.MEDIAPIPE)
        # self.processor_hmr = create_processor(ProcessorType.HAMER)
        self._register_routes()

    def _register_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/upload/image', methods=['POST'])
        def upload_image():
            file = request.files.get('file')
            if not file:
                return jsonify({"error": "No file uploaded"}), 400
            return jsonify(self.processor_mp.process_image_file(file))
        
        @self.app.route('/upload/video', methods=['POST'])
        def upload_video():
            file = request.files.get('file')
            if not file:
                return jsonify({"error": "No file uploaded"}), 400
            return jsonify(self.processor_mp.process_video_file(file))
            
    def run(self):
        self.app.run(host=self.host, port=self.port)
