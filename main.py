import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import os

def split_video_on_scenes(video_path, output_dir, threshold=30.0):
    """
    Splits a video into smaller videos at each scene cut.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the smaller videos.
        threshold (float): Scene detection threshold. Lower values detect more scenes.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize VideoManager and SceneManager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # Start scene detection
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Get scene list
    scene_list = scene_manager.get_scene_list()
    print(f"Detected {len(scene_list)} scenes.")

    # Split the video at each scene
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for i, (start_time, end_time) in enumerate(scene_list):
        start_frame = int(start_time.get_seconds() * fps)
        end_frame = int(end_time.get_seconds() * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        output_file = os.path.join(output_dir, f"scene_{i + 1:03d}.mp4")

        out = None
        frame_idx = start_frame
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if out is None:
                height, width = frame.shape[:2]
                out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

            out.write(frame)
            frame_idx += 1

        if out:
            out.release()

    cap.release()
    video_manager.release()

if __name__ == "__main__":
    video_path = "input_video.mp4"  # Replace with your video file path
    output_dir = "output_scenes"    # Replace with your desired output directory
    split_video_on_scenes(video_path, output_dir)
