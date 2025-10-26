"""
=====================================================================
 Deep Visual Feature Extractor (for Event Segmentation)
=====================================================================
Extracts deep visual embeddings from video frames using pretrained
CNN backbones from the full torchvision model zoo.

Supported Models:
---------------------------------------------------------------------
 - **All torchvision.models architectures**, including:
     • Classification: ResNet, VGG, DenseNet, EfficientNet,
       MobileNet (v2/v3), MNASNet, ShuffleNet, SqueezeNet,
       ResNeXt, Wide-ResNet, Inception, GoogLeNet, AlexNet
     • Detection: Faster R-CNN, Mask R-CNN, Keypoint R-CNN,
       RetinaNet (COCO-pretrained)
     • Segmentation: FCN, DeepLabV3, LR-ASPP (VOC/COCO-pretrained)
     • Video: R3D, MC3, R2Plus1D (Kinetics-400-pretrained)
     • Quantized models (models.quantization.*)
 - Legacy aliases (e.g., "resnet50", "vgg16", "fasterrcnn_coco")
   remain supported for backward compatibility.

How to Specify a Model:
---------------------------------------------------------------------
 - Classification:     "resnet50", "vgg16", "efficientnet_b0", "mobilenet_v3_large"
 - Detection:          "detection.fasterrcnn_resnet50_fpn", "detection.maskrcnn_resnet50_fpn"
 - Segmentation:       "segmentation.deeplabv3_resnet101", "segmentation.fcn_resnet50"
 - Video (3D models):  "video.r3d_18", "video.mc3_18", "video.r2plus1d_18"
 - Quantized:          "quantization.resnet50", "quantization.mobilenet_v2"

Outputs:
---------------------------------------------------------------------
 - *_embeddings.csv : one row per sampled frame (time, feature vector)
 - *_visualchange.csv : frame-to-frame cosine distances (semantic drift)

---------------------------------------------------------------------
 MODEL SELECTION GUIDE
---------------------------------------------------------------------
Choose the model according to your research goal, computational budget,
and the type of visual change you aim to capture.

=== CLASSIFICATION MODELS =====================================================

1. **ResNet Family (resnet18 / 34 / 50 / 101 / 152)** — *Trained on ImageNet-1K*
   - General-purpose deep feature extractors.
   - Balanced representation of objects, textures, and spatial patterns.
   - Recommended baseline for most analyses—robust and interpretable.

2. **Wide-ResNet (wide_resnet50_2 / wide_resnet101_2)** — *ImageNet-1K*
   - Wider ResNet variants with doubled bottleneck width.
   - Capture finer detail and subtle visual differences.
   - More GPU/memory intensive; ideal when discriminability matters.

3. **ResNeXt (resnext50_32x4d / resnext101_32x8d)** — *ImageNet-1K*
   - Aggregated residual transformations (multi-branch ResNet).
   - High discriminability with efficient computation.
   - Great for distinguishing subtle visual categories.

4. **VGG Family (vgg11 / 13 / 16 / 19 and BN variants)** — *ImageNet-1K*
   - Classic CNNs without residual connections—linear and interpretable.
   - Good for reproducing legacy analyses or visualizing filter hierarchies.

5. **DenseNet (densenet121 / 169 / 201 / 161)** — *ImageNet-1K*
   - Highly connected feature maps; strong performance at modest size.
   - Encodes fine-grained textural and structural details.

6. **EfficientNet Family (efficientnet_b0–b7)** — *ImageNet-1K*
   - Parameter-efficient, high-performing modern architecture.
   - Excellent trade-off between speed and accuracy.

7. **MobileNet (mobilenet_v2 / mobilenet_v3_small / mobilenet_v3_large)** — *ImageNet-1K*
   - Lightweight and fast.
   - Good for exploratory or CPU-bound analysis.

8. **MNASNet (mnasnet0_5 / 0_75 / 1_0 / 1_3)** — *ImageNet-1K*
   - NAS-designed mobile architectures.
   - Ultra-lightweight; useful for scaling across many videos.

9. **ShuffleNet (shufflenet_v2_x0_5 / 1_0 / 1_5 / 2_0)** — *ImageNet-1K*
   - Optimized for speed; efficient for real-time or embedded pipelines.

10. **SqueezeNet (squeezenet1_0 / 1_1)** — *ImageNet-1K*
    - Extremely compact model; minimal memory footprint.
    - Best for low-resource environments.

11. **Inception v3 / GoogLeNet** — *ImageNet-1K*
    - Multi-scale filters; good for complex object compositions.
    - Slightly different input size (299×299).
    - Captures diverse spatial scales.

12. **AlexNet** — *ImageNet-1K*
    - Legacy CNN; useful for comparison or pedagogical baselines.

13. **places365**
    - *Trained on Places365-Standard (8M scene images, 365 categories).*
    - Trained for *scene classification* (context, layout, setting type).
    - Ideal for detecting contextual or environmental shifts.

=== DETECTION MODELS =========================================================

14. **fasterrcnn_resnet50_fpn / fasterrcnn_mobilenet_v3_large(_320)_fpn** — *COCO train2017*
    - Object-centric models sensitive to *what* appears in the frame.
    - Ideal for detecting scene/object-change boundaries.

15. **maskrcnn_resnet50_fpn** — *COCO train2017*
    - Adds segmentation masks for objects.
    - Captures both identity and spatial extent.

16. **keypointrcnn_resnet50_fpn** — *COCO train2017 (person keypoints subset)*
    - Detects human poses and body configurations.
    - Ideal for analyzing action/motion boundaries.

17. **retinanet_resnet50_fpn** — *COCO train2017*
    - Single-stage object detector.
    - Faster inference, moderate accuracy.

=== SEGMENTATION MODELS =====================================================

18. **fcn_resnet50 / fcn_resnet101** — *COCO train2017 → Pascal VOC classes*
    - Fully Convolutional Networks for dense semantic segmentation.
    - Good for capturing global scene layout and region changes.

19. **deeplabv3_resnet50 / deeplabv3_resnet101** — *COCO train2017 → Pascal VOC classes*
    - State-of-the-art scene segmentation models.
    - Capture both semantics and fine spatial detail.
    - Best for identifying high-level contextual boundaries.

20. **lraspp_mobilenet_v3_large** — *COCO train2017 → Pascal VOC classes*
    - Lightweight segmentation model for mobile/fast applications.
    - Good for coarse scene layout changes at low cost.

=== VIDEO (3D CNN) MODELS ====================================================

21. **video.r3d_18** — *Kinetics-400 (human actions)*
    - 3D ResNet; encodes motion and spatiotemporal dynamics.
    - Ideal for action-based segmentation.

22. **video.mc3_18** — *Kinetics-400*
    - Mixed 2D/3D convolutions.
    - Balances spatial and temporal sensitivity.

23. **video.r2plus1d_18** — *Kinetics-400*
    - (2+1)D separable convolutions.
    - Very strong for detecting action-based boundaries in video stimuli.

=== QUANTIZED MODELS =========================================================

24. **quantization.resnet18 / quantization.resnet50** — *ImageNet-1K (INT8 quantized)*
    - High-speed, low-memory inference; same representations as float models.

25. **quantization.mobilenet_v2 / quantization.mobilenet_v3_large** — *ImageNet-1K (INT8)*
    - Compact, fast, and CPU-friendly models for large-scale analysis.

26. **quantization.shufflenet_v2 / quantization.resnext101_32x8d** — *ImageNet-1K (INT8)*
    - Quantized efficient variants for rapid exploratory runs.

---------------------------------------------------------------------
 ADDITIONAL NOTES
---------------------------------------------------------------------
This extractor dynamically adapts to any valid torchvision model name
using a universal model loader. It removes classification heads and
global-averages feature maps, ensuring consistent 1D embeddings across
architectures (classification, detection, segmentation, and video).

Author: Sam Ketcheson
=====================================================================
"""

import os, cv2, torch, torch.nn as nn, numpy as np, pandas as pd
from torchvision import models, transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Config
# =========================
BOUNDARY_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Analyzed_Data\Kernal\KDE_Results"
OUTPUT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\event_seg analysis\Neural_model\Frame_Embeddings"

# Automatically collect all videos except "practice_clip.mp4"
VIDEO_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Tasks\taskScripts\resources\Movie_Task\videos"
VIDEOS = [
    f for f in os.listdir(VIDEO_DIR)
    if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    and f.lower() != "practice_clip.mp4"
]

if not VIDEOS:
    raise FileNotFoundError(f"No valid video files found in {VIDEO_DIR}")
else:
    print(f"Found {len(VIDEOS)} videos: {VIDEOS}")


BOUNDARY_SUFFIX = "_consensus_boundaries.csv"
MODEL_TYPE = "fasterrcnn_coco"      # options: resnet50, fasterrcnn_coco, vgg16, vgg19, efficientnet_b0, places365, or any torchvision.models.* string
USE_BOUNDARIES = False
FPS_SAMPLE = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Universal Model Loader 
# =========================
def load_universal_model(model_type="fasterrcnn_coco"):
    """
    Load nearly any torchvision model (classification, detection, segmentation, or video)
    and return a feature extractor that outputs a 1-D embedding per frame.
    """

    # Handle legacy aliases for backward compatibility
    if model_type == "fasterrcnn_coco":
        model_type = "detection.fasterrcnn_resnet50_fpn"

    # Dynamically resolve model
    if "." in model_type:
        pkg, name = model_type.split(".")
        submod = getattr(models, pkg)
        net = getattr(submod, name)(weights="DEFAULT")
    else:
        net = getattr(models, model_type)(weights="DEFAULT")

    # Strip classifier heads if present
    if hasattr(net, "fc"): net.fc = nn.Identity()
    if hasattr(net, "classifier"): net.classifier = nn.Identity()
    if hasattr(net, "heads"): net.heads = nn.Identity()

    net.eval()

    print(f'Model = {model_type}')

    def _extract_fn(x):
        with torch.no_grad():
            y = net(x)
            if isinstance(y, dict):  # segmentation/detection outputs
                for key in ["out", "pool", "layer4"]:
                    if key in y:
                        feats = y[key]; break
                else:
                    feats = list(y.values())[-1]
            elif isinstance(y, (list, tuple)):
                feats = y[0]
            else:
                feats = y
            if feats.ndim > 2:
                feats = torch.mean(feats, dim=[2, 3])
            return feats.squeeze().cpu().numpy().flatten()

    return net, _extract_fn

# =========================
# Legacy Model Loader (kept for compatibility)
# =========================
def load_model(model_type="fastercnn_coco"):
    """Load the specified pretrained CNN backbone."""
    if model_type == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        extractor = nn.Sequential(*list(net.children())[:-1])
        feature_dim = 2048
    
    elif model_type == "wide_resnet50_2":
        net = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        extractor = nn.Sequential(*list(net.children())[:-1])
        feature_dim = 2048
        print("Loaded Wide ResNet-50-2 (2× wider bottleneck).")

    elif model_type == "wide_resnet101_2":
        net = models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V2)
        extractor = nn.Sequential(*list(net.children())[:-1])
        feature_dim = 2048
        print("Loaded Wide ResNet-101-2 (2× wider bottleneck).")

    elif model_type == "mnasnet0_5":
        net = models.mnasnet0_5(weights=models.MNASNet0_5_Weights.IMAGENET1K_V1)
        extractor = nn.Sequential(*list(net.children())[:-1])
        feature_dim = 1280
        print("Loaded MNASNet-0.5 (lightweight mobile model).")

    elif model_type == "mnasnet1_0":
        net = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1)
        extractor = nn.Sequential(*list(net.children())[:-1])
        feature_dim = 1280
        print("Loaded MNASNet-1.0 (balanced performance).")

    elif model_type == "fasterrcnn_coco":
        det = fasterrcnn_resnet50_fpn(weights="COCO_V1")
        extractor = det.backbone.body
        feature_dim = 2048

    elif model_type == "vgg16":
        net = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        extractor = nn.Sequential(*list(net.features.children()))
        feature_dim = 4096

    elif model_type == "vgg19":
        net = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        extractor = nn.Sequential(*list(net.features.children()))
        feature_dim = 4096

    elif model_type == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        extractor = nn.Sequential(*list(net.children())[:-1])
        feature_dim = 1280

    elif model_type == "places365":
        try:
            import torchvision.models as models_places
            net = models_places.resnet18(num_classes=365)
            extractor = nn.Sequential(*list(net.children())[:-1])
            feature_dim = 512
        except Exception:
            raise ImportError("Places365 model not installed. See: https://github.com/CSAILVision/places365")

    else:
        # fallback to universal loader if not explicitly listed
        print(f"Using universal loader for model: {model_type}")
        net, _ = load_universal_model(model_type)
        extractor = net
        feature_dim = None
    print(f'model = {model_type}')

    extractor.eval()
    return extractor, feature_dim


# =========================
# Continue existing code
# =========================
model, FEATURE_DIM = load_model(MODEL_TYPE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =========================
# Preprocessing
# =========================
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# =========================
# Frame grabber
# =========================
def grab_frame(video_path, time_sec):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_no = int(fps * time_sec)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# =========================
# Feature extractor
# =========================
def extract_features(x):
    with torch.no_grad():
        if MODEL_TYPE == "fasterrcnn_coco":
            feats_dict = model(x)
            if "pool" in feats_dict:
                feats = feats_dict["pool"]
            else:
                feats = feats_dict[list(feats_dict.keys())[-1]]
            feats = torch.mean(feats, dim=[2, 3])
        else:
            feats = model(x)
            if feats.ndim > 2:
                feats = torch.mean(feats, dim=[2, 3])
        return feats.squeeze().cpu().numpy().flatten()

# =========================
# Main loop (unchanged)
# =========================
for vid in VIDEOS:
    print(f'Computing Embeddings for {vid}')
    base = os.path.splitext(vid)[0]
    video_path = os.path.join(VIDEO_DIR, vid)
    boundary_file = os.path.join(BOUNDARY_DIR, f"{base}{BOUNDARY_SUFFIX}")

    if USE_BOUNDARIES and os.path.exists(boundary_file):
        df_b = pd.read_csv(boundary_file)
        if "ConsensusTime(s)" not in df_b.columns:
            print(f"No 'ConsensusTime(s)' in {boundary_file}")
            continue
        times = df_b["ConsensusTime(s)"].dropna().values
    else:
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = n_frames / fps
        times = np.arange(0, duration, 1 / FPS_SAMPLE)
        cap.release()

    feats, t_valid = [], []
    for t in times:
        img = grab_frame(video_path, t)
        if img is None:
            continue
        x = preprocess(img).unsqueeze(0).to(device)
        f = extract_features(x)
        feats.append(f)
        t_valid.append(t)

    if not feats:
        print(f"No frames processed for {vid}")
        continue

    df = pd.DataFrame(feats)
    df.insert(0, "Time(s)", t_valid)
    out_csv = os.path.join(OUTPUT_DIR, f"{base}_embeddings.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved embeddings: {out_csv}")

    diffs = [1 - cosine_similarity([feats[i]], [feats[i + 1]])[0, 0]
             for i in range(len(feats) - 1)]
    diffs.append(np.nan)
    df_change = pd.DataFrame({"Time(s)": t_valid, "VisualChange": diffs})
    out_change = os.path.join(OUTPUT_DIR, f"{base}_visualchange.csv")
    df_change.to_csv(out_change, index=False)
    print(f"Saved visual change: {out_change}")

print("All videos processed.")
