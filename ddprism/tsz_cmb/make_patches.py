"""Functions for making patches around halos."""
import os
import argparse
from functools import lru_cache

import h5py
import numpy as np
import healpy as hp

def ang2diamond(
    nside: int, theta: np.ndarray, phi: np.ndarray, nest: bool = True
) -> np.ndarray:
    """
    Given nside pixel, return the pixel IDs for parent pixel at child res.

    Arguments:
        nside: HEALPix nside.
        theta: Sky position in radians. Shape (N,).
        phi: Sky position in radians. Shape (N,).
        nest: Pixel ordering of the returned pixel IDs. Defaults
            to True.

    Returns:
        Pixel IDs at original nside resolution. Shape (N, 4).
    """

    parent_pix = hp.ang2pix(nside // 2, theta, phi, nest=nest)
    children_pix = (
        4 * parent_pix[..., None] + np.array([0,1,2,3], dtype=np.int64)[None]
    )

    if not nest:
        children_pix = hp.nest2ring(nside, children_pix)

    return children_pix


def grow_diamond(
    nside: int, theta: np.ndarray, phi: np.ndarray, num_pixels: int,
    nest: bool = True
) -> np.ndarray:
    """
    Grow a diamond around a point of size Npix x Npix pixels by graph expansion.

    The diamond is grown in L = (Npix-2)//2 steps, starting from
    the 2×2 children of the parent pixel at nside/2 that contains the point.
    Each step adds all neighbours of the current set of pixels simultaneously.
    This produces a diamond shape because each pixel has up to 8 neighbours,
    and the 2×2 seed has 4 neighbours outside the seed, which are added
    in the first step, and then each of those has 4 new neighbours outside
    the current set, etc.

    Arguments:
        nside: HEALPix nside.
        theta: Sky position in radians. Shape (N,).
        phi: Sky position in radians. Shape (N,).
        num_pixels: Desired diamond size (num_pixels x num_pixels). Must be
            even and ≥ 4.
        nest: Pixel ordering of the input map AND returned pixel IDs. Default:
            True.

    Returns:
        Pixel IDs at nside in the diamond, in sorted order. Shape
            (num_pixels**2,).
    """
    # Check valid size.
    if num_pixels < 4 or (num_pixels % 2):
        raise ValueError("Npix must be even and ≥ 4 (e.g., 4, 6, 8, ...).")

    steps = (num_pixels - 2) // 2  # number of expansion steps

    neigh = ang2diamond(nside, theta, phi, nest=nest).flatten()
    pixel_ids = set(neigh.tolist())

    # Add neighbors for the required number of steps.
    for s in range(steps):
        neigh = np.unique(
            hp.get_all_neighbours(nside, neigh, nest=nest).flatten()
        )
        pixel_ids.update(neigh[neigh >= 0].tolist())

    # Return as sorted array for reproducibility (RING/NEST IDs as requested)
    return np.array(sorted(pixel_ids), dtype=np.int64)


@lru_cache(maxsize=256)
def get_ring_pixels(nside: int, total_pixels: int) -> np.ndarray:
    """
    Get the pixel indices for a nested HEALPix map of given size.

    Arguments:
        nside: HEALPix nside.
        total_pixels: Desired total number of pixels (total_pixels).

    Returns:
        Pixel indices. Shape (total_pixels,).
    """
    return np.sort(hp.nest2ring(nside, np.arange(total_pixels)))


def reorder_diamond(nside: int, picked: np.ndarray) -> np.ndarray:
    """
    Reorder pixels to be the first n_pixels**2 pixels in the nested ordering.

    Arguments:
        nside: HEALPix nside
        picked: Pixel indices from the diamond search. Shape (n_pixels**2,).

    Returns:
        Reordered pixel indices.
    """
    ring_base = get_ring_pixels(nside, len(picked))
    ring_picked = hp.nest2ring(nside, picked)

    # This is the order I need to put picked in so that it ring_picked is sorted
    picked_ring_map = np.argsort(ring_picked)
    picked_sorted = picked[picked_ring_map]
    map_array = np.zeros_like(picked_sorted)

    # The sorted ring maps to these pixels in the original space.
    map_array[hp.ring2nest(nside, ring_base)] = picked_sorted

    return map_array


# def plot_diamond(nside, lon, lat, picked, pixel_id, m, nest=False):
#     """
#     Plot the picked pixels in various views.
#     (gnom, cart) X (before and after reordering to nested)

#     Args:
#         nside (int): HEALPix nside
#         lon (float): longitude of the center in degrees
#         lat (float): latitude of the center in degrees
#         picked (array): array of picked pixel indices from the diamond search
#         m (array): map
#         pixel_id (int): central pixel id for plotting
#     """
#     # build a masked map with only the picked pixels visible
#     M = np.full(hp.nside2npix(nside), hp.UNSEEN, dtype=float)
#     M[picked] = m[picked]

#     # reorder the picked pixels to be in a nested structure
#     fake_map, pixel_id = reorder_diamond(nside, picked, M, nest=nest)

#     # quick gnomonic cutout
#     hp.gnomview(
#         M, nest=nest, rot=(lon, lat, 0),
#         xsize=xsize*4, reso=reso,  # tweak as you like (pixels across, arcmin/pixel)
#         notext=True, cbar=True, title="Picked pixels (gnomview)"
#     )
#     plt.show()
#     # mollview
#     # hp.mollview(M, nest=False, title="Picked pixels — full-sky (Mollweide)",
#     #         notext=True, cbar=True)
#     # hp.graticule()
#     # hp.projscatter(lon, lat, lonlat=True, s=6, color="k")
#     # plt.show()

#     # -------- Plate Carrée / Cartesian (flattened) with auto-zoom --------
#     # small padding based on pixel size
#     span_deg = (xsize * reso) / 60.0
#     half = 0.5 * span_deg
#     pad_fac = 1

#     _lon = lon - 360 if lon > 180 else lon   # cartview wants -180 to 180

#     lonra = [_lon - half * pad_fac, _lon + half * pad_fac]
#     latra = [lat - half * pad_fac, lat + half * pad_fac]

#     hp.cartview(M, nest=nest,
#                 notext=True, cbar=False, lonra=lonra, latra=latra,
#                 hold=1)

#     print(lonra, latra, _lon, lat)
#     hp.projplot(_lon, lat, lonlat=True, marker="x", color="k")
#     plt.show()

#     # -------- Reorder to nested plots --------
#     theta_rad, phi_rad = hp.pix2ang(nside, pixel_id)
#     lon_deg, lat_deg = hp.pix2ang(nside, pixel_id, lonlat=True)

#     span_deg = (xsize * reso) / 60.0
#     half = 0.5 * span_deg
#     pad_fac = 1

#     _lon_deg = lon_deg - 360 if lon_deg > 180 else lon_deg   # cartview wants -180 to 180

#     lonra = [_lon_deg - half * pad_fac, _lon_deg + half * pad_fac]
#     latra = [lat_deg - half * pad_fac, lat_deg + half * pad_fac]


#     hp.gnomview(
#             fake_map, nest=False, rot=(lon_deg, lat_deg, 0),
#             xsize=xsize*4, reso=reso,  # tweak as you like (pixels across, arcmin/pixel)
#         )

#     plt.show()

#     hp.cartview(fake_map, nest=False,
#                     notext=True, cbar=False, lonra=lonra, latra=latra,
#                     hold=1)

#     plt.show()


def xyz_to_lonlat(xyz):
    """Convert Cartesian coordinates to longitude and latitude in degrees.

    Arguments:
        xyz: Cartesian coordinates. Shape (N, 3).

    Returns:
        Tuple of longitude and latitude. Shape (N,).
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    ra = np.arctan2(y, x) % (2*np.pi) # wrap to [0, 2π)
    dec = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)) # [-π/2, π/2]

    lon = np.degrees(ra)
    lat = np.degrees(dec)

    return lon, lat


def process_patch(nside: int, pos: np.ndarray, num_pixels: int) -> np.ndarray:
    """
    Get the patch indices for a diamond around (lon, lat)

    Args:
        nside: HEALPix nside.
        pos (float): XYZ position of the cluster shape (3,)
        num_pixels: Size of the diamond (num_pixels x num_pixels). Must be
            even and ≥ 4.
        nest: Pixel ordering of the returned pixel IDs. Default:
            True.

    Returns:
        Pixel indices at nside in the diamond. Shape (num_pixels**2,).
    """
    lon, lat = xyz_to_lonlat(pos)

    theta = np.radians(90.0 - lat)
    phi = np.radians(lon)

    picked = grow_diamond(nside, theta, phi, num_pixels)

    return picked

    # generate_patches(
    #     halo_pos, halo_mass, halo_id, freqs,
    #     num_pixels=num_pixels, outdir=outdir, dataset_name=dataset_name,
    #     random_pos=random_pos, noise=noise, fwhm=fwhm
    # )

def generate_patches(
    halo_pos: np.ndarray, halo_mass: np.ndarray, halo_id: np.ndarray,
    freqs: list[int], num_pixels: int, map_name: str, output_file: str,
    random_pos: bool = False
):
    """Generate patches around halos.

    Arguments:
        halo_pos: XYZ positions of the halos. Shape (N, 3).
        halo_mass: Masses of the halos. Shape (N,).
        halo_id: IDs of the halos. Shape (N,).
        freqs: Frequencies of the maps.
        num_pixels: Size of the diamond.
        map_name: Name of the map.
        output_file: Output file.
    """
    # Load maps.
    maps = []
    for f in freqs:
        maps.append(
            hp.read_map(
                map_name % f, dtype=np.float32, memmap=False, nest=True
            )
        )
    nside = hp.get_nside(maps[0])
    maps = np.stack(maps, axis=-1)

    patches = []
    masses = []
    ids = []
    vecs = []

    for h_idx, h_pos, h_mass in zip(halo_id, halo_pos, halo_mass):

        picked = process_patch(nside, h_pos, num_pixels)
        if len(picked) != num_pixels**2:
            continue
        picked = reorder_diamond(nside, picked)

        patches.append(maps[picked])
        masses.append(h_mass)
        ids.append(h_idx)
        vecs.append(np.array(hp.pix2vec(nside, picked)).T)

    with h5py.File(output_file, "w") as f:
        f.create_dataset("patches", data=np.array(patches))
        f.create_dataset("vecs", data=np.array(vecs))
        if not random_pos:
            f.create_dataset("mass", data=np.array(masses))
            f.create_dataset("id", data=np.array(ids))

        f.attrs["nside"] = int(nside)
        f.attrs["num_pixels"] = int(num_pixels)
        f.attrs["freqs"] = np.array(freqs)
        f.attrs["map_name"] = map_name


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(description="Make patches around halos.")
    parser.add_argument(
        "--profile_str", type=str, default="b16",
        help="Profile string (b16, b16g7, b16g7rel, default: b16)"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="dT_tsz",
        help="Name of the dataset (default: dT_tsz)"
    )
    parser.add_argument(
        "--random_pos", action="store_true",
        help="Use random positions instead of halo positions."
    )
    parser.add_argument(
        "--noise", type=float, default=0,
        help="Add this noise in uK to each pixel."
    )
    parser.add_argument(
        "--fwhm", type=float, default=2, help="Load maps with this fwhm beam."
    )
    parser.add_argument(
        "--seed", type=int, default=100, help="Seed underlying simulation."
    )
    parser.add_argument(
        "--numpixels", type=int, default=64,
        help="Size of the diamond (num_pixels x num_pixels)."
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory."
    )
    args = parser.parse_args()

    profile_str = args.profile_str
    dataset_name = args.dataset_name
    random_pos = args.random_pos
    noise = args.noise
    fwhm = args.fwhm
    seed = args.seed
    base_dir = '/mnt/home/abayer/ceph/fastpm/halfdome/oneweek/'
    num_pixels = args.numpixels
    outdir = args.outdir

    # Setup output directory.
    os.makedirs(outdir, exist_ok=True)

    # TODO: Hardcoded frequencies.
    freqs = [93, 143, 353]

    # Load halos.
    f_halos = (
        '/mnt/home/abayer/ceph/fastpm/halfdome/stampede2_3750Mpch_6144cube' +
        f'/final_res/halos/lightcone_{seed}.hdf5'
    )
    with h5py.File(f_halos, "r") as f:
        halo_pos = f['Position'][:]
        halo_mass = f['halo_mass_m200c'][:]

    # Apply selection (mass only for now)
    mass_cut = 2e14
    mask = halo_mass > mass_cut
    halo_pos = halo_pos[mask]
    halo_mass = halo_mass[mask]

    # Sort by mass and define new id.
    asort = np.argsort(halo_mass)[::-1]
    halo_pos = halo_pos[asort]
    halo_mass = halo_mass[asort]
    halo_id = np.arange(len(halo_mass))

    if random_pos:
        # Assign completely random (isotropic) positions of correct shape
        halo_pos = np.random.normal(size=(len(halo_pos), 3))
        halo_pos /= np.linalg.norm(halo_pos, axis=1, keepdims=True)

    # Get the dataset name to use.
    if fwhm > 0 and noise > 0:
        map_name = (
            base_dir + f'final/{profile_str}/{dataset_name}_noise{noise:.0f}'
            + f'_s{seed}_f%s_fwhm{fwhm:.0f}.fits'
        )
        output_file = os.path.join(
            outdir,
            f'{dataset_name}_patches_noise{noise:.0f}_s{seed}_fwhm{fwhm:.0f}' +
            f'_random_{random_pos}.h5'
        )
    elif fwhm > 0 and noise == 0:
        map_name = (
            base_dir + f'final/{profile_str}/{dataset_name}_s{seed}' +
            f'_f%s_fwhm{fwhm:.0f}.fits'
        )
        output_file = os.path.join(
            outdir,
            f'{dataset_name}_patches_s{seed}_fwhm{fwhm:.0f}' +
            f'_random_{random_pos}.h5'
        )
    else:
        map_name = (
            base_dir + f'final/{profile_str}/{dataset_name}_s{seed}_f%s.fits'
        )
        output_file = os.path.join(
            outdir,
            f'{dataset_name}_patches_s{seed}_random_{random_pos}.h5'
        )
    # Generate patches with noise and smoothing.
    generate_patches(
        halo_pos, halo_mass, halo_id, freqs,
        num_pixels=num_pixels, map_name=map_name, output_file=output_file,
        random_pos=random_pos
    )

if __name__ == "__main__":
    main()
