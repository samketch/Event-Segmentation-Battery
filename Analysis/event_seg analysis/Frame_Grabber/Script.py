import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
VIDEO_DIR = r"C:\Users\Smallwood Lab\friends-event-segmentation\Tasks\taskScripts\resources\Movie_Task\videos"
BOUNDARY_DIR = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Frame_Grabber"

VIDEOS = ["friends1.mp4", "friends2.mp4", "friends3.mp4"]  # update names if needed
BOUNDARY_SUFFIX = "_consensus_boundaries.csv"  # matches your KDE output files

FRAME_WIDTH = 1280   # resize for plotting
FRAME_HEIGHT = 720
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def grab_frame(video_path, time_sec):
    """Return frame (as RGB image) at given time in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_no = int(fps * time_sec)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    success, frame = cap.read()
    cap.release()
    if not success:
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

    # Collect frames
    frames = []
    for t in times:
        img = grab_frame(video_path, t)
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
