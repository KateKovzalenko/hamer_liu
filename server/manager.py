from flask import Flask, request, jsonify, render_template
from server.processors.processor_factory import ProcessorType, create_processor
import cv2
import numpy as np

class Manager:
    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
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
            return jsonify(self.processor_mp.process_video_file(file))
            
    def run(self):
        self.app.run(host=self.host, port=self.port)


"""def _register_routes(self):
        @self.app.route("/infer/hamer", methods=["POST"])
        def infer_hamer():
            if 'image' not in request.files:
                return jsonify({"error": "no image"}), 400
            f = request.files['image']
            data = f.read()
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            # optional query param: ?render=1
            render = request.args.get("render", "0") == "1"
            res = self.hamer_processor.process(img, return_image=render)
            return jsonify(res)

        @self.app.route('/upload/image', methods=['POST'])
        def upload_image():
            file = request.files.get('file')
            if not file:
                return jsonify({"error": "No file uploaded"}), 400
            
            result = self.hamer_processor.process_image_file(file)
            return jsonify(result)"""


"""
curl -X POST -F "image=@test.jpg" "http://localhost:5000/infer/hamer?render=1" -o resp.json
# if resp.json contains image_base64, decode:
jq -r '.image_base64' resp.json | base64 -d > out.png
"""


"""
Notes / caveats
Do NOT attempt to open GUI windows (cv2.imshow) inside container — return images to client, or save on host and open there.
Load heavy models once at server startup for performance and GPU memory usage.
Ensure Docker image has all runtime dependencies (detectron2, PyTorch with correct CUDA, ViTPose code). These are heavy and must be installed in your image — test locally first.
Thread-safety: if you use multiple worker processes (gunicorn), each will load models. For Flask dev server single-process is easier.
If ViTPose/detectron2 cause slow builds in Docker, iterate locally first.
"""