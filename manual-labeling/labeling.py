from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_file
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
matplotlib.use('Agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
video_path = None
recording_path = None
score_table_path = None

total_frames = 0
score_table = None
behaviors = [
    "locomoting",
    "face_grooming",
    "body_grooming",
    "rearing",
    "pausing",
]
current_frame = None
current_behavior_index = None
labeling_mode = "viewing"

# Create the upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def initialize_score_table():
    global recording_path
    global score_table_path
    global score_table
    global total_frames
    global behaviors

    score_table_path = os.path.join(recording_path, 'behavior_score.csv')
    if os.path.exists(score_table_path):
        score_table = pd.read_csv(score_table_path, index_col=0)
        behaviors = score_table.columns.tolist()
        total_frames = score_table.shape[0]
    else:
        score_table = generate_empty_score_table()
        score_table.to_csv(score_table_path)


def generate_empty_score_table():
    global total_frames
    global behaviors

    return pd.DataFrame(np.zeros((total_frames, len(behaviors))).astype(int), columns=behaviors)


def generate_toy_score_table():
    global score_table
    global total_frames
    global behaviors

    score_table = pd.DataFrame(np.zeros((total_frames, len(behaviors))).astype(int), columns=behaviors)
    frame_index = 0  # Renamed to avoid conflict with global current_frame

    while frame_index < total_frames:
        behavior_idx = np.random.randint(0, len(behaviors))

        # Determine the segment length for this behavior
        if total_frames - frame_index >= 200:
            segment_length = np.random.randint(200, min(500, total_frames - frame_index))
        else:
            segment_length = total_frames - frame_index

        # Update the score table for the current segment of frames
        score_table.iloc[frame_index:frame_index + segment_length, behavior_idx] = 1

        # Move to the next segment of frames
        frame_index += segment_length


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
    # Render the HTML template and pass the behavior list
    return render_template('index.html', behaviors=behaviors)


@app.route('/get_behaviors', methods=['GET'])
def get_behaviors():
    return jsonify(behaviors=behaviors)


@app.route('/add_behavior', methods=['POST'])
def add_behavior():
    global behaviors
    global score_table

    new_behavior = request.json.get('behavior')

    if new_behavior and new_behavior not in behaviors:
        behaviors.append(new_behavior)

        # update the score table to add a new column for the new behavior
        if score_table is not None:
            score_table[new_behavior] = 0

        else:
            generate_toy_score_table()
            score_table[new_behavior] = 0

    return jsonify(behaviors=behaviors)


@app.route('/remove_behavior', methods=['POST'])
def remove_behavior():
    global behaviors
    global score_table

    behavior_to_remove = request.json.get('behavior')

    if behavior_to_remove in behaviors:
        # idx = behaviors.index(behavior_to_remove)
        behaviors.remove(behavior_to_remove)

        # update the score table to remove the column for the behavior
        if score_table is not None:
            score_table.drop(columns=behavior_to_remove, inplace=True)

    return jsonify(behaviors=behaviors)


@app.route('/upload', methods=['POST'])
def upload_file():
    global video_path
    global recording_path
    global current_frame

    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)
        recording_path = os.path.dirname(video_path)
        extract_video_info(video_path)  # Get total frames after upload
        initialize_score_table()
        current_frame = 0  # Reset current frame to 0
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


@app.route('/update_frame', methods=['POST'])
def update_frame():
    global current_frame
    global labeling_mode
    global current_behavior_index
    global score_table

    frame_number = request.json.get('frame_number')

    if frame_number is not None:
        current_frame = frame_number

        # if labeling mode is active, update the score table for the current behavior
        if labeling_mode == "tagging" and current_behavior_index is not None:
            score_table.iloc[current_frame, current_behavior_index] = 1
        elif labeling_mode == "removing" and current_behavior_index is not None:
            score_table.iloc[current_frame, current_behavior_index] = 0

    return jsonify(success=True)


@app.route("/toggle_labeling_mode", methods=['POST'])
def toggle_labeling_mode():
    global labeling_mode

    mode = request.json.get('mode')
    if mode == "tagging":
        labeling_mode = "tagging" if labeling_mode != "tagging" else "viewing"
    elif mode == "removing":
        labeling_mode = "removing" if labeling_mode != "removing" else "viewing"

    return jsonify(labeling_mode=labeling_mode)


@app.route('/update_behavior', methods=['POST'])
def update_behavior():
    global current_behavior_index
    behavior_index = request.json.get('behavior_index')

    if behavior_index is not None:
        current_behavior_index = behavior_index

    return jsonify(success=True)


@app.route('/get_behaviors_for_frame', methods=['GET'])
def get_behaviors_for_frame():
    global score_table
    global current_frame

    if current_frame is None or score_table is None:
        return jsonify(behaviors=[])

    # Get behaviors labeled as 1 in the current frame
    labeled_behaviors = score_table.columns[score_table.iloc[current_frame] == 1].tolist()

    return jsonify(behaviors=labeled_behaviors)


@app.route('/plot_score_table')
def plot_score_table():
    global score_table
    global behaviors
    global total_frames
    global video_path
    global current_frame
    global current_behavior_index

    if video_path is None:
        # return a placeholder image
        fig, ax = plt.subplots(figsize=(10, len(behaviors) * 0.5))
        ax.imshow(np.zeros((total_frames, len(behaviors))).T, aspect='auto', cmap='Greys', interpolation='none')
        ax.set_xlabel('Frame Number')
        ax.set_yticks(range(len(behaviors)))
        ax.set_yticklabels(behaviors)
        # ax.set_title('Behavior Raster Plot')

        # shade the current behavior
        if current_behavior_index is not None:
            ax.axhspan(current_behavior_index - 0.5, current_behavior_index + 0.5, facecolor='lightgray', alpha=0.5)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close(fig)

        return send_file(img, mimetype='image/png')

    # if score_table is None:
    #     # generate a toy score table
    #     generate_toy_score_table()

    fig, ax = plt.subplots(figsize=(10, len(behaviors) * 0.5))
    ax.imshow(score_table.T, aspect='auto', cmap='Greys', interpolation='none')
    ax.set_xlabel('Frame Number')
    ax.set_yticks(range(len(behaviors)))
    ax.set_yticklabels(behaviors)
    # ax.set_title('Behavior Raster Plot')

    # shade the current behavior is set
    if current_behavior_index is not None:
        ax.axhspan(current_behavior_index - 0.5, current_behavior_index + 0.5, facecolor='lightgray', alpha=0.5)

    # add a red vertical line to indicate the current frame
    if current_frame is not None:
        ax.axvline(x=current_frame, color='red',linewidth=2, linestyle='-')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)

    return send_file(img, mimetype='image/png')


@app.route('/export_csv', methods=['GET'])
def export_csv():
    global score_table

    si = io.StringIO()
    score_table.to_csv(si, index=False)

    output = Response(si.getvalue(), mimetype='text/csv')
    output.headers['Content-Disposition'] = 'attachment; filename=behavior_score_table.csv'
    return output


@app.route('/upload_score_table', methods=['POST'])
def upload_score_table():
    global score_table
    global behaviors
    global total_frames
    global score_table_path

    if 'score_table' not in request.files:
        return redirect(url_for('index'))

    file = request.files['score_table']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        score_table_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(score_table_path)

        # Load the score table into memory
        score_table = pd.read_csv(score_table_path)
        behaviors = score_table.columns.tolist()
        total_frames = score_table.shape[0]

        return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)