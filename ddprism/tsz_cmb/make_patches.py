#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
from mpi4py import MPI

import argparse
import healpy as hp
import matplotlib.pyplot as plt

from utils import xyz_to_lonlat


#####################################

def ang2diamond(nside, theta, phi, nest=False):
    """
    Given a nside map, pick the parent pixel at nside/2 that contains
    (RA,Dec), and return the 4 child pixel IDs (a 2x2 diamond) at nside_hi.

    Parameters
    ----------
    nside : int
        map nside.
    theta, phi : float
        Sky position in degrees, in the map’s coordinate frame.
    nest : bool
        Pixel ordering of the input map AND returned pixel IDs.

    Returns
    -------
    children : np.ndarray (shape (4,), dtype=int)
        Pixel IDs at nside.
    """
    
    nside_hi = nside
    nside_lo = nside // 2

    # Parent at lower NSIDE in the requested ordering
    parent_lo = hp.ang2pix(nside_lo, theta, phi, nest=nest)

    # Convert to NESTED to enumerate children easily
    if nest:
        parent_nest = parent_lo
    else:
        parent_nest = hp.ring2nest(nside_lo, parent_lo)

    # The 4 children in NESTED at the higher NSIDE
    children_nest_hi = 4*parent_nest + np.array([0,1,2,3], dtype=np.int64)

    # Convert back to requested ordering
    if nest:
        children_hi = children_nest_hi
    else:
        children_hi = hp.nest2ring(nside_hi, children_nest_hi)

    return children_hi

def grow_diamond(nside, theta_deg, phi_deg, Npix, nest=False):
    """
    Grow a diamond around a point of size Npix x Npix pixels by graph expansion.
    The diamond is grown in L = (Npix-2)//2 steps, starting from
    the 2×2 children of the parent pixel at nside/2 that contains the point.
    Each step adds all neighbours of the current set of pixels simultaneously.
    This produces a diamond shape because each pixel has up to 8 neighbours,
    and the 2×2 seed has 4 neighbours outside the seed, which are added
    in the first step, and then each of those has 4 new neighbours outside
    the current set, etc.
    
    Parameters
    ----------
    nside : int
        Map nside.
    theta_deg, phi_deg : float
        Sky position in degrees, in the map’s coordinate frame.
    Npix : int  
        Desired diamond size (Npix x Npix pixels). Must be even and ≥ 4.
    nest : bool
        Pixel ordering of the input map AND returned pixel IDs. Default: False (RING).
    Returns
    -------
    S : np.ndarray (shape (M,), dtype=int)
        Pixel IDs at nside in the diamond, in sorted order.
    """
    if Npix < 4 or (Npix % 2):
        raise ValueError("Npix must be even and ≥ 4 (e.g., 4, 6, 8, ...).")
    L = (Npix - 2) // 2  # number of expansion steps
    seeds = ang2diamond(nside, theta_deg, phi_deg, nest=nest)
    S = set(int(p) for p in seeds)
    getN = hp.get_all_neighbours

    for _ in range(L):
        # snapshot of current set to expand this step (simultaneous growth)
        cur = list(S)
        additions = set()
        for p in cur:
            neigh = getN(nside, p, nest=nest)
            for nb in neigh:
                if nb >= 0:
                    additions.add(int(nb))
        S |= additions

    # return as sorted array for reproducibility (RING/NEST IDs as requested)
    return np.array(sorted(S), dtype=np.int64)

def reorder_diamond(nside, picked, M, nest=False):
    """
    Reorder picked pixels to be in a nested structure by mapping to ring and back to 0... nest.
    
    Args:
        nside (int): HEALPix nside
        picked (array): array of picked pixel indices from the diamond search
        M (array): masked map with only picked pixels (1 otherwise)
        nest (bool): whether the input picked pixels are in nested ordering (default: False)
    
    Returns:
        array: reordered picked pixel indices in ring ordering 
    
    """

    fake_map = np.full(hp.nside2npix(nside), hp.UNSEEN, dtype=float)

    parallel = hp.nest2ring(nside, np.arange(len(picked)))
    fake_map[np.sort(parallel)] = M[np.sort(picked)]

    pixel_id = np.sort(parallel)[len(parallel)//2]   # this is just useful for plotting center

    return fake_map, pixel_id

def plot_diamond(nside, lon, lat, picked, pixel_id, m, nest=False):
    """
    Plot the picked pixels in various views.
    (gnom, cart) X (before and after reordering to nested)
    
    Args:
        nside (int): HEALPix nside
        lon (float): longitude of the center in degrees
        lat (float): latitude of the center in degrees
        picked (array): array of picked pixel indices from the diamond search
        m (array): map 
        pixel_id (int): central pixel id for plotting   
    """
    # build a masked map with only the picked pixels visible
    M = np.full(hp.nside2npix(nside), hp.UNSEEN, dtype=float)
    M[picked] = m[picked]
    
    # reorder the picked pixels to be in a nested structure
    fake_map, pixel_id = reorder_diamond(nside, picked, M, nest=nest)

    # quick gnomonic cutout
    hp.gnomview(
        M, nest=nest, rot=(lon, lat, 0),
        xsize=xsize*4, reso=reso,  # tweak as you like (pixels across, arcmin/pixel)
        notext=True, cbar=True, title="Picked pixels (gnomview)"
    )
    plt.show()
    # mollview
    # hp.mollview(M, nest=False, title="Picked pixels — full-sky (Mollweide)",
    #         notext=True, cbar=True)
    # hp.graticule()
    # hp.projscatter(lon, lat, lonlat=True, s=6, color="k")
    # plt.show()

    # -------- Plate Carrée / Cartesian (flattened) with auto-zoom --------
    # small padding based on pixel size
    span_deg = (xsize * reso) / 60.0
    half = 0.5 * span_deg
    pad_fac = 1
    
    _lon = lon - 360 if lon > 180 else lon   # cartview wants -180 to 180
    
    lonra = [_lon - half * pad_fac, _lon + half * pad_fac]
    latra = [lat - half * pad_fac, lat + half * pad_fac]

    hp.cartview(M, nest=nest,
                notext=True, cbar=False, lonra=lonra, latra=latra,
                hold=1)
    
    print(lonra, latra, _lon, lat)
    hp.projplot(_lon, lat, lonlat=True, marker="x", color="k")
    plt.show()

    # -------- Reorder to nested plots --------
    theta_rad, phi_rad = hp.pix2ang(nside, pixel_id)
    lon_deg, lat_deg = hp.pix2ang(nside, pixel_id, lonlat=True)

    span_deg = (xsize * reso) / 60.0
    half = 0.5 * span_deg
    pad_fac = 1

    _lon_deg = lon_deg - 360 if lon_deg > 180 else lon_deg   # cartview wants -180 to 180

    lonra = [_lon_deg - half * pad_fac, _lon_deg + half * pad_fac]
    latra = [lat_deg - half * pad_fac, lat_deg + half * pad_fac]


    hp.gnomview(
            fake_map, nest=False, rot=(lon_deg, lat_deg, 0),
            xsize=xsize*4, reso=reso,  # tweak as you like (pixels across, arcmin/pixel)
        )

    plt.show()

    hp.cartview(fake_map, nest=False,
                    notext=True, cbar=False, lonra=lonra, latra=latra,
                    hold=1)

    plt.show()
    
###################

def process_patch(nside, pos, Npix, nest=False):
    """
    Process a patch around (lon, lat) into a Npix x Npix diamond.
    This includes growing the diamond, building the masked map,
    reordering to nested, and plotting.
    
    Args:
        m: the map being processed
        pos (float): xyz position of the cluster shape (3,)
        Npix: size of diamond
        
        nest (bool): whether the input map is in nested ordering (default: False)
        plot (bool): whether to plot the results (default: False)
    """
    
    assert(nest==False)  # only RING input supported for now
    
    lon, lat = xyz_to_lonlat(pos)
    
    theta = np.radians(90 - lat)
    phi = np.radians(lon)
    
    picked = grow_diamond(nside, theta, phi, Npix, nest=nest)
        
    return picked

#######################

def main(halo_pos, halo_mass, noise=0, fwhm=0, outdir="./patches_out", dataset_name="blah", random_pos=False):
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size

    if rank == 0:
        os.makedirs(outdir, exist_ok=True)
    
    # load maps
    if fwhm > 0 and noise > 0:
        f_map = mother + f'final/{profile_str}/{dataset_name}_noise{noise:.0f}_s{seed}_f%s_fwhm{fwhm:.0f}.fits'
    elif fwhm > 0 and noise == 0:
        f_map = mother + f'final/{profile_str}/{dataset_name}_s{seed}_f%s_fwhm{fwhm:.0f}.fits'
    else:
        f_map = mother + f'final/{profile_str}/{dataset_name}_s{seed}_f%s.fits'
    
    maps = np.empty((len(freqs), hp.nside2npix(nside)), dtype=np.float32)   # store all maps in memory for easy reading out later
    for i, f in enumerate(freqs):
        maps[i] = hp.read_map(f_map % f, dtype=np.float32, memmap=False)

    assert(nside == hp.get_nside(maps[0]))

    # Split work (round-robin so all ranks get similar count)
    N = len(halo_pos)
    idx_all = np.arange(N)
    idx_my  = np.array_split(idx_all, size)[rank]   # rank 0 gets [0:...], rank 1 gets next, etc.
    N_local = idx_my.size

    # Early exit if nothing to do
    if N_local == 0:
        # still barrier to let others finish
        comm.Barrier()
        return

    out_vecs = np.empty((N_local, Npix**2, 3), dtype=np.float32)  # shape (Npix*Npix, 3)
    out_vals = np.empty((N_local, Npix**2, len(freqs)), dtype=np.float32)  # shape (Npix*Npix, Nfreq)
    if not random_pos:
        out_mass = np.empty((N_local,), dtype=np.float32)
        out_ids = np.empty((N_local,), dtype=np.int32)
        
    for j, i in enumerate(idx_my):
        if j % 500 == 0: print(rank, j, flush=True)
        
        picked = process_patch(nside, halo_pos[i], Npix, nest=False)
        
        # FIXME some clusters near boundareis on healpix dont build full diamonds (very rare)
        # this causes errors. For now, completely drop these clusters.
        if len(picked) != Npix**2:
            out_vecs[j] = np.full(out_vecs.shape[1:], np.nan)
            out_vals[j] = np.full(out_vals.shape[1:], np.nan)
            if not random_pos:
                out_mass[j] = halo_mass[i]
                out_ids[j] = halo_id[i]
            continue
    
        # we want to match sorted(picked) to sorted(parallel) and then order according to nexted ordering (matches what happens in reorder_diamond())
        parallel = hp.nest2ring(nside, np.arange(len(picked)))
        swap = hp.ring2nest(nside, np.sort(parallel))   # this is the nest id in the mapped space of each pixel in np.sort(picked)
        asort = np.argsort(swap)
        pixs = np.sort(picked)[asort]
        
        vals = np.empty((len(pixs), len(freqs)), dtype=np.float32)
        vecs = hp.pix2vec(nside, pixs)
        
        for fi,freq in enumerate(freqs):
            vals[:, fi] = maps[fi,pixs]
    
        out_vecs[j] = np.asarray(vecs, dtype=np.float32).T
        out_vals[j] = np.asarray(vals, dtype=np.float32)
        if not random_pos:
            out_mass[j] = halo_mass[i]
            out_ids[j] = halo_id[i]

    # ---- write once per rank ----
    random_str = "random" if random_pos else "halo"
    shard = os.path.join(outdir, f"{dataset_name}_{random_str}patches_noise{noise:.0f}_fwhm{fwhm:.0f}.{rank:d}.h5")
    with h5py.File(shard, "w") as f:
        #comp = dict(compression="gzip", compression_opts=4, shuffle=True)
        f.create_dataset("vals", data=out_vals)#, **comp)    # (N_local, P, F)
        f.create_dataset("vecs", data=out_vecs)#, **comp)    # (N_local, P, 3)
        if not random_pos:
            f.create_dataset("mass", data=out_mass)#,   **comp)    # (N_local,)
            f.create_dataset("id", data=out_ids)#,   **comp)    # (N_local,)
        f.attrs["nside"] = int(nside)
        f.attrs["Npix"] = int(Npix)
        f.attrs["freqs"] = np.array(freqs, dtype=np.float32)
        f.attrs["dataset_name"] = dataset_name

    comm.Barrier()
    if rank == 0:
        print("done writing shards ->", outdir)

if __name__ == "__main__":
    # Toggle gzip by setting compress=True if disk is tight (slower writes).
    
    # argarse
    # make  into arg
    parser = argparse.ArgumentParser(description="Make patches around halos.")
    parser.add_argument("--profile_str", type=str, default="b16", help="Profile string (b16, b16g7, b16g7rel, default: b16)")
    parser.add_argument("--dataset_name", type=str, default="dT_tsz", help="Name of the dataset (default: dT_tsz)")
    parser.add_argument("--random_pos", action="store_true", help="Use random positions instead of halo positions.")
    parser.add_argument("--noise", type=float, default=0, help="Add this noise in uK to each pixel.")
    parser.add_argument("--fwhm", type=float, default=2, help="Load maps with this fwhm beam.")
    args = parser.parse_args()

    profile_str = args.profile_str
    dataset_name = args.dataset_name
    random_pos = args.random_pos
    noise = args.noise
    fwhm = args.fwhm

    ### INPUT ##########
    seed = 100
    mother = '/mnt/home/abayer/ceph/fastpm/halfdome/oneweek/'
    Npix = 64  # diamond size of patch
    nside = 8192   # FIXME assumed can change later

    outdir =  f'/mnt/ceph/users/abayer/fastpm/halfdome/oneweek/final/{profile_str}/patches/'   # FIXME change
    os.makedirs(outdir, exist_ok=True)
    
    #### GLOBALS ############

    freqs = [93, 143, 353]

    f_halos = f'/mnt/home/abayer/ceph/fastpm/halfdome/stampede2_3750Mpch_6144cube/final_res/halos/lightcone_{seed}.hdf5'   

    with h5py.File(f_halos, "r") as f:
        halo_pos = f['Position'][:]
        halo_mass = f['halo_mass_m200c'][:]

    # apply selection (mass only for now)
    mass_cut = 2e14
    mask = halo_mass > mass_cut
    halo_pos = halo_pos[mask]
    halo_mass = halo_mass[mask]

    # sort by mass and define new id for easy debugging
    asort = np.argsort(halo_mass)[::-1]
    halo_pos = halo_pos[asort]
    halo_mass = halo_mass[asort]
    halo_id = np.arange(len(halo_mass))

    # FIXME remove just debugging
    # halo_pos = halo_pos[:100]
    # halo_mass = halo_mass[:100]
    # halo_id = halo_id[:100]
    
    if random_pos:
        # assign completely random (isotropic) positions of correct shape
        halo_pos = np.random.normal(size=(len(halo_pos), 3))
        halo_pos /= np.linalg.norm(halo_pos, axis=1, keepdims=True)
    
    main(halo_pos, halo_mass, noise=noise, fwhm=fwhm, outdir=outdir, dataset_name=dataset_name, random_pos=random_pos)
