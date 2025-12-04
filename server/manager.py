from flask import Flask, request, jsonify, render_template
from server.processors.processor_factory import ProcessorType, create_processor

class Manager:
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        # Configuration
        self.app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024 
        self.app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
        self.app.config['JSON_AS_ASCII'] = False
        self.app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
        
        self.hamer_processor = create_processor(ProcessorType.HAMER)
        self._register_routes()

    def _parse_bool(self, value):
        """Helper to convert request values to boolean."""
        if not value:
            return False
        return str(value).lower() in ['true', '1', 't', 'y', 'yes', 'on']

    def _register_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/upload/image', methods=['POST'])
        def upload_image():
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "No file uploaded"}), 400
            
            # PARSING FIX: Convert string form-data to boolean
            person_selector = request.form.get("person_selector", "all")
            hand_side = request.form.get("hand_side", "both")
            should_render = self._parse_bool(request.form.get("should_render"))

            result = self.hamer_processor.process_image_file(
                file, 
                person_selector=person_selector, 
                hand_side=hand_side,
                should_render=should_render
            )
            return jsonify(result)
        
        @self.app.route('/upload/video', methods=['POST'])
        def upload_video():
            file = request.files.get('file')
            if not file:
                return jsonify({"error": "No file uploaded"}), 400
            
            # PARSING FIX: Convert string form-data to boolean
            person_selector = request.form.get("person_selector", "all")
            hand_side = request.form.get("hand_side", "both")
            should_render = self._parse_bool(request.form.get("should_render"))
            
            try:
                result = self.hamer_processor.process_video_file(
                    file,
                    person_selector=person_selector,
                    hand_side=hand_side,
                    should_render=should_render
                )
            except Exception as e:
                import traceback
                tb_str = traceback.format_exc()
                print(tb_str)
                return jsonify({"error": str(e), "traceback": tb_str}), 500
            return jsonify(result)
            
    def run(self):
        self.app.run(host=self.host, port=self.port)
