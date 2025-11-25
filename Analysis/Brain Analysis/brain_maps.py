import os
import nibabel as nib
from nilearn import plotting

GRADIENT_DIR = r"C:\Users\Smallwood Lab\Documents\Event-Segmentation-Battery\Analysis\Brain Analysis\State Space\Gradients"
OUTPUT_DIR   = os.path.join(GRADIENT_DIR, "Figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(1, 6):
    path = os.path.join(GRADIENT_DIR, f"volume.313.{i}.nii")
    img = nib.load(path)

    plotting.plot_glass_brain(
        img,
        display_mode='lyrz',
        colorbar=True,
        title=f"Gradient {i}",
        cmap='coolwarm',
        threshold=None,
    ).savefig(os.path.join(OUTPUT_DIR, f"gradient_{i}_glass.png"), dpi=300)

    plotting.plot_stat_map(
        img,
        title=f"Gradient {i}",
        cmap='coolwarm',
        display_mode='ortho',
        cut_coords=(0, 0, 0),
        colorbar=True,
    ).savefig(os.path.join(OUTPUT_DIR, f"gradient_{i}_orth.png"), dpi=300)

print("✅ Saved gradient brain figures to:", OUTPUT_DIR)
