# Image Datasets for Benchmarking ParametricDFT

A literature survey of image datasets suitable for benchmarking the parametric quantum Fourier transform algorithm for image compression and sparse representation.

## Overview

Given that ParametricDFT learns adaptive transforms for image compression using frequency-dependent truncation, we need datasets that cover:
1. **Smooth images** - Natural photographs with gradual tonal transitions
2. **Linework images** - Cartoons, manga, sketches with sharp edges and sparse strokes

---

## Part 1: Smooth/Natural Image Datasets

### 1.1 Kodak PhotoCD Dataset

**Description**: The classic benchmark for image compression, consisting of 24 uncompressed true-color images.

| Property | Value |
|----------|-------|
| Size | 24 images |
| Resolution | 768 × 512 pixels |
| Format | PNG (lossless) |
| Content | Diverse natural scenes, portraits, objects |

**Example Image** (kodim23 - Parrots):

![Kodak kodim23 - Parrots](https://raw.githubusercontent.com/lemire/kodakimagecollection/master/kodim23.png)

**Why suitable**:
- Industry standard for compression benchmarks
- Smooth gradients and natural textures
- Well-characterized baseline results available
- Small enough for quick iteration

**URL**: http://r0k.us/graphics/kodak/

**Reference**: Kodak lossless true color image suite (PhotoCD PCD0992)

---

### 1.2 DIV2K Dataset

**Description**: Large-scale high-resolution dataset with diverse natural content, widely used for super-resolution and compression research.

| Property | Value |
|----------|-------|
| Training | 800 images |
| Validation | 100 images |
| Test | 100 images |
| Resolution | 2K resolution (~2048 pixels on long edge) |

**Example**: DIV2K contains diverse 2K images including people, nature, urban scenes, flora/fauna, and underwater shots. Preview available via [Hugging Face dataset viewer](https://huggingface.co/datasets/eugenesiow/Div2k/viewer/).

**Why suitable**:
- High resolution allows testing at multiple scales
- Diverse content (people, nature, urban, objects)
- Standard benchmark for learned compression methods
- Smooth natural images with varying frequency content

**URL**: https://data.vision.ee.ethz.ch/cvl/DIV2K/

**Reference**: Agustsson, E. and Timofte, R. "NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study"

---

### 1.3 CLIC Dataset (Challenge on Learned Image Compression)

**Description**: Professional and mobile image datasets from the annual learned compression challenge.

| Property | Value |
|----------|-------|
| Professional Train | 585 images |
| Professional Valid | 41 images |
| Mobile Valid | 61 images |
| Resolution | Various (high resolution) |

**Example**: High-quality professional photographs (768×768 for validation). Preview available on the [CLIC challenge page](https://compression.cc/).

**Why suitable**:
- Specifically designed for compression benchmarks
- Professional subset has high-quality natural photos
- Mobile subset has smartphone photos (different characteristics)
- Active competition provides comparison baselines

**URL**: https://www.compression.cc/

**Reference**: CLIC Workshop (annual since 2018)

---

### 1.4 Tecnick TESTIMAGES

**Description**: High-resolution test images specifically designed for evaluating compression algorithms.

| Property | Value |
|----------|-------|
| Size | 100 images (40 commonly used subset at 1200×1200) |
| Resolution | 1200 × 1200 (common benchmark subset) |
| Content | Natural photographs |

**Example**: High-resolution natural photographs for compression benchmarking. Browse at [testimages.org/sampling](https://testimages.org/sampling/).

**Why suitable**:
- Designed specifically for compression testing
- High resolution with good detail
- Widely used in lossless and lossy compression research

**URL**: https://testimages.org/

**Reference**: Asuni, N. and Giachetti, A. "TESTIMAGES: A Large-scale Archive for Testing Visual Devices and Basic Image Processing Algorithms" (2014)

---

### 1.5 Classic SR Benchmarks (Set5, Set14, BSD100, Urban100)

**Description**: Classic small-scale benchmarks commonly used together for super-resolution and image reconstruction.

| Dataset | Size | Content |
|---------|------|---------|
| Set5 | 5 images | Classic test images (baby, bird, butterfly, head, woman) |
| Set14 | 14 images | Mix of human, animal, natural scenes (baboon, barbara, lenna, zebra, etc.) |
| BSD100 | 100 images | Natural images from Berkeley Segmentation |
| Urban100 | 100 images | Urban scenes with repetitive patterns |

**Example Image** (Set5 - Butterfly):

![Set5 Butterfly](https://raw.githubusercontent.com/idealo/image-super-resolution/master/figures/butterfly.png)

**Why suitable**:
- Quick evaluation during development
- Urban100 has high-frequency repetitive structures (tests edge preservation)
- BSD100 derived from BSDS500 with segmentation ground truth
- Standard in the literature, enabling comparison

**Reference**:
- Set5: Bevilacqua et al. (2012)
- Set14: Zeyde et al. (2012)
- BSD100: Martin et al. (2001)
- Urban100: Huang et al. (2015)

**Download**: Available at [LapSRN project page](https://github.com/jbhuang0604/SelfExSR)

---

### 1.6 BSD68 (Denoising Benchmark)

**Description**: 68-image subset of Berkeley Segmentation Dataset, standard for denoising evaluation.

| Property | Value |
|----------|-------|
| Size | 68 images |
| Content | Natural scenes |
| Usage | Typically tested with Gaussian noise (σ = 15, 25, 50) |

**Example**: Natural scene images from the Berkeley Segmentation Dataset. Original BSDS available at [Berkeley BSDS page](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/).

**Why suitable**:
- Standard for sparse coding and dictionary learning methods
- Well-characterized for PSNR/SSIM baselines
- Tests reconstruction quality on natural images

**Reference**: Martin et al. "A Database of Human Segmented Natural Images" (2001)

---

## Part 2: Linework/Cartoon Image Datasets

### 2.1 Manga109 Dataset

**Description**: Large manga dataset with professional Japanese comic pages, widely used as SR/compression benchmark.

| Property | Value |
|----------|-------|
| Volumes | 109 manga titles |
| Pages | 21,142 total pages |
| Annotations | Frames, text, faces, bodies (500k+ annotations) |
| Commercial subset | 87 volumes (Manga109-s) |

**Example**: Professional B&W manga pages with annotated frames, speech bubbles, and character faces. See examples in the [Manga109 arXiv paper (Fig. 1-4)](https://arxiv.org/abs/2005.04425).

![Manga109 Sample](https://ar5iv.labs.arxiv.org/html/2005.04425/assets/x1.png)

**Why suitable**:
- **Standard benchmark** for super-resolution alongside natural image sets
- Pure black-and-white line art with sharp edges
- High contrast, sparse strokes - tests edge preservation
- Mix of detailed and simple art styles
- Academic use freely available

**URL**: http://www.manga109.org/en/

**Reference**: Matsui, Y. et al. "Sketch-based manga retrieval using Manga109 dataset" (2017)

---

### 2.2 BIPED Dataset (Barcelona Images for Perceptual Edge Detection)

**Description**: Urban scene images with carefully annotated edge ground truth.

| Property | Value |
|----------|-------|
| Size | 250 images |
| Resolution | 1280 × 720 pixels |
| Annotations | Expert-annotated edge maps |

**Example**: Urban scenes from Barcelona with expert-annotated structural edges.

![BIPED Banner](https://github.com/xavysp/MBIPED/raw/master/figs/BIPED_banner.png)

**Why suitable**:
- Ground truth edges enable quantitative edge preservation metrics
- Urban scenes have both smooth regions and sharp edges
- Can evaluate how well transform preserves structural edges

**URL**: https://xavysp.github.io/MBIPED/

**Reference**: Poma, X.S. et al. "Dense Extreme Inception Network for Edge Detection" (Pattern Recognition, 2023)

---

### 2.3 ATD-12K (Anime Triplet Dataset)

**Description**: Large-scale anime frame triplets from professional animation studios.

| Property | Value |
|----------|-------|
| Training | 10,000 triplets |
| Test | 2,000 triplets |
| Source | 30 animation movies (~25 hours total) |
| Annotations | Difficulty levels, motion types, RoI |

**Example**: Professional anime frames from various animation studios (Disney, Japanese anime).

![ATD-12K Sample](https://raw.githubusercontent.com/lisiyao21/AnimeInterp/main/figs/sample0.png)

**Why suitable**:
- Professional anime quality
- Color cartoon images with flat colors and sharp outlines
- Variety of animation styles
- Large scale enables training-based methods

**URL**: https://github.com/lisiyao21/AnimeInterp

**Reference**: Siyao, L. et al. "Deep Animation Video Interpolation in the Wild" (CVPR 2021)

---

### 2.4 iCartoonFace Dataset

**Description**: Large-scale cartoon face dataset with identity and attribute annotations.

| Property | Value |
|----------|-------|
| Images | 389,678 |
| Characters | 5,013 |
| Annotations | Identity, bounding box, pose, attributes |

**Example**: Cartoon character faces from various animated series with different styles.

![iCartoonFace Challenges](https://ar5iv.labs.arxiv.org/html/1907.13394/assets/challenge.jpg)

**Why suitable**:
- Largest cartoon image dataset available
- Consistent cartoon style faces
- Tests transform on stylized content with flat colors

**Reference**: Zheng, Y. et al. "Cartoon Face Recognition: A Benchmark Dataset" (2019)

---

### 2.5 Danbooru Dataset

**Description**: Large-scale anime illustration dataset with extensive tagging.

| Property | Value |
|----------|-------|
| Size | 4M+ images |
| Tags | Extensive multi-label annotations |
| Content | Anime/manga style illustrations |

**Example**: Diverse anime-style illustrations ranging from clean line art to fully colored/shaded artwork. Preview available via [Gwern's Danbooru2021 page](https://www.gwern.net/Danbooru2021).

**Why suitable**:
- Massive scale for training
- Diverse anime/illustration styles
- Mix of clean line art and painted styles

**Note**: Requires careful filtering for research-appropriate content.

---

### 2.6 QuickDraw Dataset

**Description**: Google's dataset of 50 million hand-drawn sketches across 345 categories.

| Property | Value |
|----------|-------|
| Sketches | 50 million |
| Categories | 345 |
| Format | Vector (stroke sequences) and rasterized |

**Example**: Simple hand-drawn sketches (cat, bicycle, tree, house, etc.) created in 20 seconds or less.

![QuickDraw Preview](https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/preview.jpg)

**Why suitable**:
- Pure sparse line drawings
- Extremely minimal strokes
- Tests compression on maximally sparse content

**URL**: https://quickdraw.withgoogle.com/data

---

## Recommended Benchmark Suite for ParametricDFT

Based on the algorithm's characteristics (frequency-dependent truncation, O(log² N) parameters), here's a recommended benchmark suite:

### Tier 1: Essential (Quick Evaluation)
| Category | Dataset | Size | Purpose |
|----------|---------|------|---------|
| Smooth | **Kodak** | 24 | Industry standard, quick iteration |
| Smooth | **Set5 + Set14** | 19 | Ablation studies |
| Linework | **Manga109** | 109 volumes | Standard SR benchmark, pure B&W |

### Tier 2: Comprehensive
| Category | Dataset | Size | Purpose |
|----------|---------|------|---------|
| Smooth | **DIV2K** | 900 | Large-scale natural images |
| Smooth | **CLIC** | ~700 | Compression-specific benchmark |
| Mixed | **Urban100** | 100 | High-frequency patterns |
| Linework | **ATD-12K** | 12,000 triplets | Color anime/cartoon |

### Tier 3: Specialized
| Category | Dataset | Size | Purpose |
|----------|---------|------|---------|
| Smooth | **Tecnick** | 100 | High-resolution compression |
| Edges | **BIPED** | 250 | Edge preservation evaluation |
| Sketch | **QuickDraw** | 50M | Extreme sparsity test |

---

## Evaluation Metrics

For comprehensive benchmarking, measure:

1. **Quality metrics**: PSNR, SSIM, MS-SSIM
2. **Perceptual metrics**: LPIPS, FID (if applicable)
3. **Rate-distortion**: Quality vs compression ratio curves
4. **Edge-specific**: Edge preservation score using BIPED ground truth
5. **Sparsity**: L0/L1 norm of transform coefficients

---

## References

1. Kodak PhotoCD: http://r0k.us/graphics/kodak/
2. DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/
3. CLIC: https://www.compression.cc/
4. Tecnick: https://testimages.org/
5. Manga109: http://www.manga109.org/en/
6. BIPED/DexiNed: https://github.com/xavysp/DexiNed
7. ATD-12K: https://github.com/lisiyao21/AnimeInterp
8. Urban100: https://github.com/jbhuang0604/SelfExSR
