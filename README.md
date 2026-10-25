# AI Predator Early Warning

A camera-trap image classifier for Aotearoa New Zealand. A six-class model (BIRD, CAT, EMPTY, HEDGEHOG, POSSUM, RODENT) is turned into a binary predator alert, and the alert is then checked for bias.

## Dataset

- **Wellington Camera Traps**, a citizen-science project in Wellington, NZ (Anton et al., 2018). Download it from TAIAO: https://datasets.cms.waikato.ac.nz/taiao/wellington_camera_traps/ (licence: Community Data License Agreement, permissive).
- 270,450 images with 17 labels. MOUSE, RAT, Ship rat and Norway rat were merged into RODENT, NOTHINGHERE became EMPTY, and the other labels were dropped.
- The data were split **by camera site** (150 / 32 / 33 sites) so that test sites are never seen in training. 3,000 images were sampled per class (2,100 / 450 / 450). After removing 37 very dark images by hand, 17,963 images remain.
- The images are not included in this repository.

## Model

- **Baseline:** a small CNN with three convolutional blocks.
- **Main model:** ImageNet-pretrained ResNet18, fully fine-tuned for 10 epochs on images resized to 1,000 px (shorter side), with mixed precision and batch size 2.
- **Alert policy:** alert score = sum of the CAT, HEDGEHOG, POSSUM and RODENT probabilities. The threshold (0.70) was chosen on the validation set only.

## Results

| Metric                                           | Value                 |
| ------------------------------------------------ | --------------------- |
| Baseline CNN, best validation accuracy           | 40.9%                 |
| ResNet18, best validation accuracy               | 71.1%                 |
| ResNet18, test six-class accuracy                | 70.4%                 |
| Test alert precision / recall / F1               | 0.947 / 0.942 / 0.945 |
| False alerts / missed alerts (2,695 test images) | 95 / 104              |

Bias check on the test set:

| Group                     | Alert recall | False alert rate |
| ------------------------- | ------------ | ---------------- |
| Colour images             | 60.6%        | 2.4%             |
| Greyscale infrared images | 96.8%        | 22.6%            |

The model partly uses infrared lighting, not only the animal, to decide that a predator is present.

## Quick start (no training)

The image cache (`data/cache_1000/`, about 5 GB) is not included because of its size. Without it, you can still:

- **Read the results:** both notebooks are saved with their outputs, and the result tables and figures are in `outputs/results_v2/` and `outputs/figures_v2/`.
- **Reproduce the main results:** open `notebooks/02_train_and_evaluate.ipynb` from the `notebooks/` folder and run **Section 0, Section 1.2, Section 2.3, then Sections 5 and 6**. These sections read the saved predictions, so no images, model or GPU are needed.

Section 4.6 (loading the trained model) and Section 7 (error examples, Grad-CAM and the bias check) need the image cache and the model weights `outputs/models/best_resnet18_high_res_v2.pth`. To rebuild the cache, follow the full pipeline below.

## How to run (full pipeline)

1. Download the images and the metadata CSV/JSON from the TAIAO page. Put the CSV and JSON in `data/raw/`. The notebook reads the images from a ZIP archive (`wct_images.zip`).
2. In `notebooks/01_prepare_dataset.ipynb`, set `ZIP_PATH` to your ZIP file, then run all cells. This creates `data/interim/final_clean_metadata.csv` and the image folders.
3. Run `notebooks/02_train_and_evaluate.ipynb` from top to bottom. After a kernel restart, run Section 0 first.
4. Optional: run `python src/bias_audit.py` from the project root to repeat the bias check.

Notes:

- Training needs an NVIDIA GPU (about 20 minutes per ResNet18 epoch on an RTX 5060 Laptop GPU).
- Predictions are cached in `outputs/results_v2/`. If you retrain the model, delete `validation_predictions.csv` and `test_predictions.csv` first.

## Libraries

Python 3.10+ and:

```
torch  torchvision  pandas  numpy  scikit-learn  matplotlib  pillow  tqdm  jupyter
```

## Limitations and future work

- **Lighting and background:** recall is much lower for colour images than for infrared images. Future work: crop animals with a detector such as MegaDetector before classifying, and add more colour images of predators.
- **Missing species:** only 27 mustelid images, so stoats cannot be reliably detected, and pet cats cannot be told apart from feral cats. Future work: add data from other datasets or thermal cameras, and keep a human check before any action.
- **One city only:** all data come from Wellington and only 33 test sites, and accuracy varies between sites. Future work: test the model in other regions and report results per site.
