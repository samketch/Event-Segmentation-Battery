import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
VIDEO_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Tasks\taskScripts\resources\Movie_Task\videos"
BOUNDARY_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Frame_Grabber"

VIDEOS = [
    f for f in os.listdir(VIDEO_DIR)
    if f.endswith(".mp4") and f.lower() != "practice_clip.mp4"
]
BOUNDARY_SUFFIX = "_consensus_boundaries.csv"  # matches your KDE output files

FRAME_WIDTH = 1280   # resize for plotting
FRAME_HEIGHT = 720
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def grab_frame(video_path, time_sec):
    """Return frame (as RGB image) at given time in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video:{video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    time_sec = max(0, min(time_sec, duration - 0.05))
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    success, frame = cap.read()
    # If it failed, try reading sequentially (slower, but works)
    if not success or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, f = cap.read()
            if not ret:
                break
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if current_time >= time_sec:
                frame = f
                success = True
                break
    cap.release()
    
    if not success or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

for vid in VIDEOS:
    video_path = os.path.join(VIDEO_DIR, vid)
    base = os.path.splitext(vid)[0]
    boundary_file = os.path.join(BOUNDARY_DIR, f"{base}{BOUNDARY_SUFFIX}")

    if not os.path.exists(boundary_file):
        print(f"Boundary file missing: {boundary_file}")
        continue

    # Load boundary times
    df = pd.read_csv(boundary_file)
    if "ConsensusTime(s)" not in df.columns:
        print(f"No 'ConsensusTime(s)' column in {boundary_file}")
        continue
    times = df["ConsensusTime(s)"].dropna().values
    print(len(times))

    # Collect frames
    frames = []
    for t in times:
        img = grab_frame(video_path, t)
        if img is None:
            print(f"Failed to grab frame at {t:.2f}s in {vid}")
        if img is not None:
            img_resized = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
            frames.append((t, img_resized))

    if not frames:
        print(f"No frames extracted for {vid}")
        continue

    # Plot storyboard-style timeline
    fig, ax = plt.subplots(1, len(frames), figsize=(len(frames) * 2, 3))
    if len(frames) == 1:
        ax = [ax]  # make iterable

    for i, (t, img) in enumerate(frames):
        ax[i].imshow(img)
        ax[i].axis("off")
        ax[i].set_title(f"{int(t//60)}:{int(t%60):02d}", fontsize=12)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, f"{base}_timeline.png")
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved: {outpath}")
