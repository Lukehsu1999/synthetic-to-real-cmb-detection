# Synthetic CMB Generation

This directory contains the synthetic cerebral microbleed (CMB)
generation code used in our synthetic-to-real CMB detection experiments.

The generator inserts parameterized 3D hypointense lesions into
**CMB-negative SWAN MRI** volumes. Synthetic lesion properties are
sampled from predefined distributions, with lesion size, elongation, and
signal contrast informed by measurements from real CMBs in the training
cohort.

The entire synthetic CMB generation pipeline is illustrated below:
![Synthetic CMB Pipeline](https://github.com/Lukehsu1999/synthetic-to-real-cmb-detection/blob/main/figures/synthetic_CMB_generation_flow.png)

> **Important:** The parameter values reported below correspond to the
> generator used in our experiments. If you modify them, the resulting
> synthetic data will no longer correspond exactly to the synthetic data
> distribution described in the paper.



## Scripts

Two versions of the generator are provided:

### `generate_single_variant_synthetic_cmb.py`

Generates **one synthetic image per healthy host image**.

This is the version corresponding to the main synthetic dataset used in
our experiments, where each CMB-negative SWAN MRI was converted into one
synthetic-positive scan.

Default:

``` text
5 lesions × 1 variant per host
```

### `generate_multi_variant_synthetic_cmb.py`

Generates **multiple independently sampled variants from each healthy
host image**.

Each variant starts from the original host MRI and independently
resamples lesion locations, sizes, shapes, orientations, contrasts, and
blending parameters.

The script currently defaults to:

``` text
3 lesions × 3 variants per host
```

Both values can be changed from the command line.

This version can be useful for increasing synthetic sample diversity
without requiring additional CMB-negative host scans.

------------------------------------------------------------------------

## Generation Pipeline

For each synthetic lesion, the generator:

1.  Samples lesion size, elongation, signal contrast, edge smoothness,
    and blending strength.
2.  Constructs a volume-preserving ellipsoidal lesion.
3.  Applies a random 3D rotation.
4.  Uniformly samples an insertion site from an eroded brain mask.
5.  Estimates local background intensity around the insertion site.
6.  Creates a soft sigmoid lesion boundary.
7.  Blends the hypointense lesion into the host SWAN MRI.
8.  Thresholds the soft alpha mask to obtain the corresponding binary
    lesion annotation.

This procedure is repeated sequentially until the requested number of
lesions has been inserted.

------------------------------------------------------------------------

## Before Running

The generator assumes that preprocessing has already been performed.

### 1. CMB-negative SWAN MRI

The host images should be **3D CMB-negative SWAN MRI volumes** stored as
`.nii` or `.nii.gz`.

The generator does **not** perform:

-   skull stripping;
-   registration;
-   resampling;
-   bias-field correction;
-   intensity normalization.

Therefore, input images should already be in the preprocessing state
appropriate for synthesis.

### 2. Brain masks

Every host image requires a corresponding brain mask.

For example:

``` text
healthy_images/
├── case001_0000.nii.gz
├── case002_0000.nii.gz
└── case003_0000.nii.gz

brain_masks/
├── case001.nii.gz
├── case002.nii.gz
└── case003.nii.gz
```

The scripts support filenames both with and without the nnU-Net `_0000`
channel suffix when searching for the corresponding mask.

The brain mask is converted internally using:

``` python
brain_mask = brainmask > 0
```

The image and mask are assumed to have matching:

-   dimensions;
-   voxel grids;
-   orientation;
-   spatial correspondence.

**No automatic mask resampling or affine-alignment check is performed.**

### 3. Correct voxel spacing

Lesion geometry is defined in physical units (mm).

Voxel spacing is obtained directly from:

``` python
header.get_zooms()[:3]
```

The NIfTI header must therefore contain correct physical voxel spacing.

### 4. Compatible SWAN intensity scale

Synthetic lesion intensity is calculated directly from the intensity
values of the input SWAN MRI.

The code does not normalize image intensities before synthesis.

The default signal-contrast distribution and the final lesion-intensity
clipping range were selected for the SWAN images used in this study. If
another dataset has a substantially different intensity distribution,
these parameters should be recalibrated.

------------------------------------------------------------------------

## Usage

### Single Variant

``` bash
python generate_single_variant_synthetic_cmb.py \
    --healthy_dir /path/to/healthy_images \
    --brainmask_dir /path/to/brain_masks \
    --output_image_dir /path/to/output/images \
    --output_mask_dir /path/to/output/labels \
    --num_lesions 5 \
    --seed 42
```

The default number of lesions is:

``` text
--num_lesions 5
```

This approximately matches the median CMB burden observed in our
real-positive training cohort.

### Multiple Variants

``` bash
python generate_multi_variant_synthetic_cmb.py \
    --healthy_dir /path/to/healthy_images \
    --brainmask_dir /path/to/brain_masks \
    --output_image_dir /path/to/output/images \
    --output_mask_dir /path/to/output/labels \
    --num_lesions 3 \
    --variants_per_brain 3 \
    --seed 42
```

For example:

``` text
--num_lesions 3
--variants_per_brain 3
```

generates three independent synthetic versions of each host MRI, with
three lesions inserted into each version.

To reproduce the single-variant lesion burden while generating multiple
versions of each host, use:

``` text
--num_lesions 5
--variants_per_brain 3
```

------------------------------------------------------------------------

## Arguments

  -----------------------------------------------------------------------
  Argument                            Description
  ----------------------------------- -----------------------------------
  `--healthy_dir`                     Directory containing CMB-negative
                                      SWAN MRI volumes

  `--brainmask_dir`                   Directory containing corresponding
                                      brain masks

  `--output_image_dir`                Output directory for synthetic MRI
                                      volumes

  `--output_mask_dir`                 Output directory for synthetic
                                      lesion masks

  `--num_lesions`                     Number of synthetic CMBs inserted
                                      into each generated scan

  `--variants_per_brain`              Number of independently generated
                                      versions of each host;
                                      multi-variant script only

  `--seed`                            NumPy random seed
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Synthetic Lesion Parameters

Each lesion independently samples its parameters in:

``` python
sample_mcb_params()
```

The current implementation uses:

  Parameter                    Sampling                         Range
  ---------------------------- -------------------------------- ------------------
  Equivalent diameter $d$      $\mathcal{N}(3.6, 0.8^2)$ mm     2.0--6.0 mm
  Elongation ratio $q$         $\mathcal{N}(1.6, 0.35^2)$       1.0--2.5
  Signal contrast $c$          $\mathcal{N}(160, 30^2)$         90--230
  Edge smoothness              $\mathcal{N}(0.30, 0.10^2)$ mm   0.15--0.60 mm
  Blending strength $s$        $\mathcal{N}(0.95, 0.05^2)$      0.85--1.00
  Orientation                  Random 3D rotation               ---
  Synthetic lesion intensity   $I_{\mathrm{bg}} - c$            Clipped to 0--60
  Binary mask                  $\alpha > 0.5$                   Fixed

Samples from the Gaussian distributions are clipped to their
corresponding ranges.

Lesion size, elongation, and signal contrast were selected with
reference to measurements from real CMBs in the **real-positive training
cohort**. The hold-out test cohort was not used to parameterize the
generator.

------------------------------------------------------------------------

## Changing the Parameter Ranges

To change the synthetic lesion distribution, modify:

``` python
def sample_mcb_params():
```

For example, lesion diameter is currently sampled using:

``` python
diameter_mm = np.random.normal(loc=3.6, scale=0.8)
diameter_mm = float(np.clip(diameter_mm, 2.0, 6.0))
```

Therefore:

``` text
loc=3.6       → Gaussian mean
scale=0.8     → Gaussian standard deviation
2.0           → minimum allowed diameter
6.0           → maximum allowed diameter
```

For example, a hypothetical generator targeting larger lesions could
use:

``` python
diameter_mm = np.random.normal(loc=4.5, scale=1.0)
diameter_mm = float(np.clip(diameter_mm, 2.5, 8.0))
```

The same structure is used for `ratio`, `contrast`, `edge_sigma_mm`, and
`strength`.

### Lesion burden

The number of lesions does not require changing the source code:

``` bash
--num_lesions 5
```

### Dataset multiplicity

For the multi-variant generator:

``` bash
--variants_per_brain 3
```

controls how many independently synthesized scans are produced from each
original host.

------------------------------------------------------------------------

## Lesion Geometry

Synthetic CMBs are modeled as randomly oriented **axisymmetric
ellipsoids**.

The sampled equivalent diameter $d$ first defines a target volume
equivalent to a sphere:

$$
V = \frac{4}{3}\pi\left(\frac{d}{2}\right)^3.
$$

Given the sampled elongation ratio $q$, the ellipsoid radii are defined
as:

$$
r_x=r_y=r,\qquad r_z=qr,
$$

with:

$$
r=\left(\frac{3V}{4\pi q}\right)^{1/3}.
$$

This separates two properties:

-   $d$ controls overall lesion size;
-   $q$ controls elongation.

Changing elongation therefore does not change the target lesion volume.

A random 3D rotation matrix is subsequently sampled to vary lesion
orientation.

------------------------------------------------------------------------

## Insertion Location

Lesion centers are uniformly sampled from the supplied brain mask after
binary erosion.

The current implementation uses:

``` python
margin_vox=8
```

The erosion reduces insertion close to the brain boundary and helps
avoid boundary artifacts.

If erosion leaves no valid voxels, the original, non-eroded brain mask
is used as a fallback.

No empirical anatomical location distribution is imposed.

This was a deliberate design choice in our study: real CMBs were
observed across much of the brain, and uniform insertion reduced
reliance on fixed spatial priors during detector training.

------------------------------------------------------------------------

## Local Background and Lesion Intensity

For every sampled insertion site, the generator estimates local
background intensity $I_{\mathrm{bg}}$.

The median intensity is calculated within a spherical shell of:

``` text
inner radius = 3 mm
outer radius = 6 mm
```

around the lesion center, restricted to voxels inside the brain mask.

The synthetic lesion intensity is then:

$$
I_{\mathrm{lesion}} = I_{\mathrm{bg}} - c,
$$

where $c$ is the sampled signal contrast.

The implementation subsequently clips this value to:

``` text
0–60
```

If fewer than 10 valid voxels are available in the 3--6 mm shell, the
implementation falls back to brain voxels in the local crop. If those
are also unavailable, it uses the median intensity of the entire image.

------------------------------------------------------------------------

## Soft Lesion Boundary

The ellipsoidal geometry is converted into a soft alpha mask using a
sigmoid boundary:

``` python
alpha = 1.0 / (
    1.0 + np.exp(
        signed_mm / max(edge_sigma_mm, 1e-6)
    )
)
```

The sampled `edge_sigma_mm` controls the softness of the lesion
boundary.

The alpha mask is subsequently restricted to the brain:

``` python
alpha = alpha * brain_crop.astype(np.float32)
```

------------------------------------------------------------------------

## Image Blending

The sampled blending strength $s$ scales the alpha mask:

$$a = s\alpha.$$

The synthetic lesion is inserted using alpha compositing:

$$I_{\mathrm{syn}} = (1-a)I_{\mathrm{host}} + aI_{\mathrm{lesion}}.$$

This produces a gradual transition between the hypointense synthetic lesion and surrounding tissue.

The corresponding binary ground-truth mask is obtained independently from the **unscaled** alpha mask:

$$\alpha > 0.5.$$

Therefore, blending strength changes the appearance of the synthetic lesion but does not directly change the threshold used to define its binary annotation.

------------------------------------------------------------------------

## Output

### Single-variant generator

For an input such as:

``` text
case001_0000.nii.gz
```

the output is:

``` text
images/
└── case001_0000.nii.gz

labels/
└── case001.nii.gz
```

This follows the image/label naming convention expected by nnU-Net-style
datasets.

### Multi-variant generator

The current multi-variant script uses:

``` text
images/
├── case001_synth502_v00_0000.nii.gz
├── case001_synth502_v01_0000.nii.gz
└── case001_synth502_v02_0000.nii.gz

labels/
├── case001_synth502_v00.nii.gz
├── case001_synth502_v01.nii.gz
└── case001_synth502_v02.nii.gz
```

The `synth502` component is retained from the experimental dataset
naming convention used during development and can be renamed if a
different public-facing naming convention is preferred.

Synthetic images are saved as `float32`; binary masks are saved as
`uint8`.

The original image affine and NIfTI header are retained.

------------------------------------------------------------------------

## Reproducibility

Use:

``` bash
--seed 42
```

to initialize NumPy's random number generator.

The random seed controls:

-   lesion parameter sampling;
-   lesion center selection;
-   random 3D orientation.

Input files are processed in sorted filename order.

For the multi-variant script, variants are also generated sequentially.
Consequently, changing the number/order of input images, number of
lesions, or number of variants changes subsequent positions in the
random sequence even when the same initial seed is used.

------------------------------------------------------------------------

## Modeling Assumptions and Limitations

The generator intentionally uses a relatively simple and controllable
lesion model.

Its main assumptions are:

1.  **Ellipsoidal geometry** --- Synthetic CMBs are approximated as
    axisymmetric ellipsoids rather than reproducing arbitrary
    real-lesion morphology.
2.  **Uniform spatial sampling** --- Lesion locations are sampled
    uniformly within the eroded brain mask. No anatomical or
    disease-specific spatial prior is modeled.
3.  **Local intensity model** --- Synthetic lesion intensity is
    determined relative to the median local background signal rather
    than by explicitly modeling the complete lesion intensity
    distribution.
4.  **Fixed parameter distributions** --- Lesion properties are sampled
    independently from predefined distributions rather than from a
    learned joint generative model.
5.  **Sequence-specific intensity assumptions** --- The contrast and
    intensity ranges were designed for the SWAN MRI data used in this
    project and may not directly transfer to differently normalized data
    or other MRI sequences.
6.  **No surrounding anatomical deformation** --- Synthesis modifies
    local voxel intensities but does not model changes to surrounding
    anatomy.

These simplifications are intentional: the generator was designed to
test whether controllable synthetic lesion supervision can transfer to
real CMB detection rather than to perfectly reproduce the full real CMB
distribution.

When adapting the generator to another dataset, we recommend validating
at minimum:

-   lesion size distribution;
-   lesion elongation distribution;
-   lesion-to-background contrast;
-   visual appearance across representative cases;
-   compatibility with the dataset's voxel spacing and intensity scale.

------------------------------------------------------------------------

## Citation

If you use this generator, please cite the accompanying paper.

**Paper citation:** *To be added upon public release.*
