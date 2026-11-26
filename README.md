# Paper Conservation Analysis

A desktop application for comparing images of paper (with ink) and reporting visible differences between reference and comparison states. Designed for paper conservation purposes.

Developed by Dr Michael Joyce at the Centre for Print Research (CFPR), University of the West of England (UWE), Bristol, UK.

## Overview

This app is designed for a simple conservation workflow:

- Choose one image for individual analysis, one before and one after, two before and one after, or one before and two after; select and preview the images, then click **Load images**
- Review and adjust masks for paper and ink regions
- Check image alignment visually
- Configure the options on the **Analysis** tab and click **Analyse**
- View the data report in **Results data**; **Image results** is reserved for future image outputs

Image labels follow the selected configuration: Image; B and A; B1, B2 and A; or B, A1 and A2.

## Features

- Load and preview one image or a set of before and after images
- Generate paper and ink masks using luminance or reference-colour modes
- Inspect mask overlays and alignment previews
- Analyse RGB and LAB colour statistics for one image or compare before and after images
- Report RGB Euclidean and redmean colour distances and CIELAB ΔE values for mean colours
- Use a second before or after image to estimate variation between replicate measurements
- Optionally compare using masks and combined masks
- Generate a readable analysis summary for each comparison

## Requirements

- Python 3.10+
- PySide6
- NumPy
- SciPy
- Pillow

## Installation

1. Clone the repository.
2. Create and activate a virtual environment if desired.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

If a requirements file is not present, install the packages manually:

```bash
pip install PySide6 numpy scipy pillow
```

## Running the app

From the project root:

```bash
python main.py
```

## Project structure

- `main.py` — desktop app entry point and UI
- `src/paperdiff/` — image loading, analysis, and workflow logic
- `src/paperdiff/image_state.py` — shared image and analysis state objects

## Notes

This project assumes the images have already been colour corrected and aligned and cropped before they are loaded into the application (e.g. using Auto-Align Layers in Photoshop). The app focuses on the comparison and reporting workflow rather than image preparation.

For a side with two images, colour averages and standard deviations are averaged and reported ± half their absolute difference. The change uses the same replicate variation. Mean-colour distances are calculated for each before–after pair and reported as their average ± half their absolute difference. With one image on each side, no replicate-based ± is reported. RGB distances use the report's 0–100 channel scale; redmean weights are calculated from the mean red channel on the conventional 0–255 scale. RGB Euclidean and redmean distances are not CIELAB ΔE values and should not be compared directly with them.

Single-image analysis reports mean colours and standard deviations for the whole image or its masked paper and ink regions. It does not report before/after changes or colour distances; combined masks are unavailable.

Selecting a new image or changing the image configuration requires loading the selected set again before masking, alignment, or analysis. Changing configurations clears the middle image selection because its role may have changed. Starting a load clears previously loaded images, masks, combined masks, previews, and the previous report while retaining selected image paths and settings.

On the Mask tab, enable **Use mask** to import or export masks. If no mask has been generated or imported, exporting produces a solid 50% grey image. Exported masks are PNG files, defaulting to the image's folder and filename with `_mask.png` appended to the filename stem; **Set export path** changes the destination until the next load. White pixels represent paper, black pixels ink, 50% grey (RGB 128, 128, 128) both, and red neither. On import, only exact white, black, and 50% grey pixels are included; all other colours are excluded. Imported masks must have the same resolution as their image.

**Export all masks** saves each image's mask to its configured path, prompting before overwriting. **Auto-import masks** loads files found at those paths and reports missing or invalid masks.

Image loading runs in the background. The Load images tab estimates progress at the midpoint of each loading step, including decoding, RGB preparation, CIELAB conversion, mask initialisation, and preview preparation. **Cancel loading** stops between loading steps; selected paths are retained, but previously loaded data has already been cleared. Individual decoding or array operations cannot be interrupted mid-step. Closing the window waits for loading to finish.

Analysis runs in the background. The Analysis tab shows the current task and estimates progress at each task's midpoint, including individual image and region comparisons; tasks may take different amounts of time. Image and analysis settings cannot be changed until it finishes. Closing the window during analysis waits for the current run to finish.

Use **Cancel analysis** to stop a run. Cancellation takes effect between calculation steps, and the previous report is preserved.
