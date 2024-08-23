from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
video_path = None
total_frames = 0

# Create the upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def extract_video_info(video_path):
    global total_frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

def generate_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    if success:
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        return frame
    cap.release()
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global video_path
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)
        extract_video_info(video_path)  # Get total frames after upload
        return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    global video_path
    frame = int(request.args.get('frame', 0))
    if video_path:
        frame_data = generate_frame(video_path, frame)
        if frame_data:
            return Response(frame_data, mimetype='image/jpeg')
    return "", 204  # No content

@app.route('/get_video_info')
def get_video_info_route():
    global video_path
    if video_path:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return jsonify(total_frames=total_frames, fps=fps)
    else:
        return jsonify(total_frames=0, fps=0)

if __name__ == "__main__":
    app.run(debug=True)
