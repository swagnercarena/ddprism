# tSZ-CMB Component Separation

This subpackage uses diffusion-based posterior inference (ddprism) to separate thermal Sunyaev-Zel'dovich (tSZ) signal from CMB+noise in multi-frequency observations.

## Data overview

All data lives on CephFS under `/mnt/home/abayer/ceph/fastpm/halfdome/`.

### Halo catalog

```
/mnt/home/abayer/ceph/fastpm/halfdome/stampede2_3750Mpch_6144cube/final_res/halos/lightcone_{seed}.hdf5
```

- Default seed: `100`
- HDF5 datasets: `Position` (Cartesian xyz, shape `(N, 3)`), `halo_mass_m200c` (solar masses)
- We apply a mass cut of `2e14` M_sun and sort descending by mass

Reading example:
```python
import h5py
with h5py.File('/mnt/home/abayer/ceph/fastpm/halfdome/stampede2_3750Mpch_6144cube/final_res/halos/lightcone_100.hdf5', 'r') as f:
    pos = f['Position'][:]       # shape (N, 3), Cartesian unit vectors
    mass = f['halo_mass_m200c'][:] # shape (N,), in solar masses
```

### HEALPix maps (input FITS files)

Base directory:
```
/mnt/home/abayer/ceph/fastpm/halfdome/oneweek/final/{profile_str}/
```

Profile strings encode the tSZ pressure profile model:
- `b16` -- baseline Battaglia 2016
- `b16g7` -- Battaglia 2016 with gamma=7
- `b16g7rel` -- Battaglia 2016 with gamma=7 and relativistic corrections (currently primary)

#### Map types

| Name | Description | Has noise | Frequencies |
|------|-------------|-----------|-------------|
| `T_tot` | Total observed temperature (CMB + tSZ + noise) | Yes | 93, 143, 353 GHz |
| `dT_tsz` | tSZ-only temperature decrement (no noise, no CMB) | No | 93, 143, 353 GHz |

#### File naming conventions

Multi-frequency maps (one FITS file per frequency):
```
# With noise and beam smoothing:
{base_dir}/final/{profile_str}/{dataset_name}_noise{noise}_s{seed}_f{freq}_fwhm{fwhm}.fits
# Example: T_tot_noise7_s100_f93_fwhm2.fits

# Without noise, with beam:
{base_dir}/final/{profile_str}/{dataset_name}_s{seed}_f{freq}_fwhm{fwhm}.fits
# Example: dT_tsz_s100_f93_fwhm2.fits

# No beam, no noise:
{base_dir}/final/{profile_str}/{dataset_name}_s{seed}_f{freq}.fits
```

Frequencies are `93`, `143`, `353` (GHz). Default beam FWHM is `2` arcmin. Default noise is `7` uK.

Reading a map:
```python
import healpy as hp
import numpy as np
m = hp.read_map(
    '/mnt/home/abayer/ceph/fastpm/halfdome/oneweek/final/b16g7rel/T_tot_noise7_s100_f93_fwhm2.fits',
    dtype=np.float32, nest=True, memmap=False
)
nside = hp.get_nside(m)  # resolution parameter
```

All maps use **NEST** pixel ordering (not RING).

### Patches (HDF5 output)

Patches are diamond-shaped cutouts of the HEALPix sphere centered on halo positions (or random sky positions for the baseline). They are produced by `make_patches.py` and stored as HDF5.

Output directory:
```
/mnt/home/abayer/ceph/fastpm/halfdome/oneweek/final/{profile_str}/patches/
```

File naming:
```
{dataset_name}_patches[_noise{noise}]_s{seed}[_fwhm{fwhm}]_random_{True|False}.h5
```

Examples:
- `T_tot_patches_noise7_s100_fwhm2_random_False.h5` -- observed data at halo positions
- `T_tot_patches_noise7_s100_fwhm2_random_True.h5` -- observed data at random positions (baseline)
- `dT_tsz_patches_s100_fwhm2_random_False.h5` -- noise-free tSZ signal at halo positions

#### HDF5 structure

```python
import h5py
import numpy as np

with h5py.File('T_tot_patches_noise7_s100_fwhm2_random_False.h5', 'r') as f:
    patches = f['patches'][:]  # shape (N_halos, num_pixels**2, n_freqs) e.g. (N, 4096, 3)
    vecs = f['vecs'][:]        # shape (N_halos, num_pixels**2, 3) -- unit vectors per pixel
    mass = f['mass'][:]        # shape (N_halos,) -- only present when random=False
    ids = f['id'][:]           # shape (N_halos,) -- only present when random=False

    # Attributes
    nside = f.attrs['nside']        # int, HEALPix resolution
    num_pixels = f.attrs['num_pixels']  # int, e.g. 64 (patches are 64x64)
    freqs = f.attrs['freqs']        # array, e.g. [93, 143, 353]
    map_name = f.attrs['map_name']  # str, template path of source maps
```

- `patches` pixel dimension is `num_pixels**2` (default 4096 = 64x64) and has been reordered into a standard nested ordering via `reorder_diamond()`
- `vecs` contains 3D unit vectors from `healpy.pix2vec()` for each pixel -- used as positional information by the transformer model
- To reshape patches into a 2D image: `patch_2d = patches[i].reshape(num_pixels, num_pixels, n_freqs)`

## Training data flow

1. **Normalization**: patches are divided by `map_norm = 2000.0` and clipped to `[-1, 1]`
2. **Noise model**: Gaussian with sigma = `7.0 / 2000.0` per pixel per frequency
3. **Mixing matrix**: identity-like (each frequency channel is an independent observation of the same sky)
4. **Two sources**: the posterior jointly infers CMB (source 0) and tSZ (source 1); `x_post` is split in half along the last axis

## Key scripts

| Script | Purpose |
|--------|---------|
| `make_patches.py` | Extract diamond patches from HEALPix maps at halo/random positions |
| `train_sz.py` | Train joint posterior denoiser (EM: Gaussian init -> diffusion) |
| `train_randoms.py` | Train baseline diffusion prior on random-position patches |
| `load_datasets.py` | Load and reshape patch HDF5 files for JAX training |
| `config_randoms.py` / `config_sz.py` | ML-collections configs for training |
