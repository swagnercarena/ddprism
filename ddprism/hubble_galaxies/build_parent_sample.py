"""Script for building the HST Cosmos parent sample.

Example usage:
    python3 build_parent_sample.py --metadata_path=metadata_main.csv
    --morphology_path=gz_hubble_main.csv --downloads_folder=COSMOS/
    --output_dir=COSMOS/ --debug=True --nan_tolerance=0.2 --zero_tolerance=0.2
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import functools
import os

from astropy.io import fits
from astropy.wcs import WCS
from astroquery.mast import Observations
import h5py
import healpy as hp
import numpy as np
import pandas as pd
from tqdm import tqdm

import cutout_utils

#####################
#### Set Globals ####
#####################
# properties of our HST images
PIXEL_SCALE = 0.05 # HST drizzled pixel scale in arcseconds
NUMPIX = 128
CUTOUT_SIZE = NUMPIX * PIXEL_SCALE # approx 100 pixel cutouts.
PIDS = ['10092', '9822']
HEALPIX_NSIDE = 128
NUM_EXPOSURES = 4 # number of exposures added together by drizzle.
DARK_CURRENT = 0.0168 # e-/s/pixel,from exposure time calculator

# COSMOS query info
# HST proposal IDs
BAND = 'F814W'
PID_LIST = [9822, 10092]


def fill_target_lists():
    """ Queries a list of tile names for each proposal ID

    NOTE: FOR 10092 TILE LIST GENERATION
        - 'COSMOS18-11A' 'COSMOS18-11B'
        - B has a big/bright object
        - 'COSMOS33-08A' 'COSMOS33-08B'
            - B has a big/bright object
        - 'COSMOS09-15' and 'COSMOS09-15A'
            - 09-15A has a big/bright object

        - COSMOS30-08: 2 entries, take one with longer exposure time
        - COSMOS30-07: 2 entries, take one with longer exposure time

    NOTE: When downloading, tile COSMOS31-19 consistently has problems
    (empty/corrupted .fits file). Currently removed it to avoid issues.
    """

    target_lists = {}

    # PID 9822 tiles can all be kept.
    obs_table = Observations.query_criteria(
        proposal_id=PID_LIST[0], filters=BAND, provenance_name='CALACS',
        dataproduct_type='image'
    )
    target_lists[str(PID_LIST[0])] = np.asarray(obs_table['target_name'])

    # PID 10092 tiles need to be pruned.
    obs_table = Observations.query_criteria(
        proposal_id=PID_LIST[1], filters=BAND, provenance_name='CALACS',
        dataproduct_type='image'
    )
    # Remove duplicates (removes copies of COSMOS30-08, COSMOS30-07).
    unique_targets = np.unique(np.asarray(obs_table['target_name']))
    # Remove COSMOS18-11B, COSMOS33-08B, COSMOS09-15A
    unique_targets = unique_targets[
        ~np.isin(
            unique_targets, ['COSMOS18-11B', 'COSMOS33-08B', 'COSMOS09-15A']
        )
    ]
    target_lists[str(PID_LIST[1])] = unique_targets

    # remove COSMOS31-19 (see note)
    for pid in ['9822','10092']:
        target_lists[pid] = target_lists[pid][
            ~np.isin(target_lists[pid], ['COSMOS31-19'])
        ]

    return target_lists


###############################################
#### Stores Catalog Info & Creates Cutouts ####
###############################################
class CosmosCutouts():
    """
    Class for generating cutouts from the COSMOS field.

    Args:
        metadata_main_path (string): path to .csv containing galaxy zoo
            source catalog. Downloaded from
            https://data.galaxyzoo.org/#section-11.
        morphology_path (string): path to .csv containing galaxy zoo
            morphological classifications
        downloads_folder (string): where to store .fits files downloaded
            from MAST
        nan_tolerance (float): What fraction of the cutout is allowed to be
            a nan. If 1.0, all values can be nans.
        zero_tolerance (float): What fraction of the cutout is allowed to be
            zero. If 1.0 all values can be zeros.
    """

    def __init__(
        self, metadata_main_path, morphology_path, downloads_folder,
        output_dir,nan_tolerance=1.0, zero_tolerance=1.0
    ):
        # set properties
        self.downloads_folder = downloads_folder
        self.nan_tolerance = nan_tolerance
        self.zero_tolerance = zero_tolerance
        self.output_dir = output_dir

        # load galaxy-zoo metadata
        metadata_main_df = pd.read_csv(metadata_main_path, low_memory=False)
        cosmos_df = metadata_main_df[metadata_main_df['imaging'] == 'COSMOS']

        # read in morphology classifications
        morph_df = pd.read_csv(morphology_path, low_memory=False)
        # set index to zooniverse ID, index in order of cosmos_df zooniverse
        # IDs, then reset index
        morph_df = (
            morph_df.set_index('zooniverse_id').loc[
                cosmos_df['zooniverse_id']
            ].reset_index()
        )
        # add in relevant morphology parameters from morph_df
        cosmos_df = cosmos_df.merge(
            morph_df, on='zooniverse_id', how='left', suffixes=(None, '_gz')
        )

        # remove rows that are flagged as an artifact & save to object
        self.cosmos_df = (
            cosmos_df[cosmos_df[
                't01_smooth_or_features_a03_star_or_artifact_flag'
            ] == False]
        )


    def make_cutouts(
        self, targets_df, flux_tile, weight_tile, wcs_tile, print_status=False
    ):
        """
        Args:
            targets_df (pd.dataframe): contains 'RA' 'DEC' for targets
            flux_tile (np.array[float], size=(npix,npix)): image for full tile.
            weight_tile (np.array[float], size=(npix,npix)): weight for full
                tile.
            wcs_tile (astropy.wcs.WCS): wcs of the image / weights tile.
            print_status (bool): If true will print the success rate.
        Returns:
            (np.array, np.array, np.array, np.array): The flux, weights, masks,
                and indices for each galaxy in the DataFrame.
        """
        # keep track of skips.
        target_idxs = targets_df.index.to_list()
        num_objects = len(target_idxs)

        # loop through targets & create the cutout
        results = map(
            functools.partial(
                cutout_utils.make_cutout_single, targets_df=targets_df,
                flux_tile=flux_tile, weight_tile=weight_tile,
                wcs_tile=wcs_tile, nan_tolerance=self.nan_tolerance,
                zero_tolerance=self.zero_tolerance, cutout_size=CUTOUT_SIZE,
                dark_current=DARK_CURRENT,num_exposures=NUM_EXPOSURES
            ),
            target_idxs
        )

        # remove the skips.
        results = [res for res in results if res is not None]
        flux_maps, ivar_maps, mask_maps, idx = (
            map(list, zip(*results)) if results else ([], [], [], [])
        )

        if print_status:
            print(
                f'Generated cutouts with a success rate of '
                f'{len(flux_maps) / num_objects:.3f}'
            )

        return flux_maps, ivar_maps, mask_maps, idx

    @staticmethod
    def _extract_numpy(df_column):
        """
        Helper function for dealing with masking in pandas to hdf5.

        Args:
            df_column (pd.DataFrame): Dataframe column to be convereted to numpy.

        Returns:
            Numpy array with NaN values properly dealt with.
        """
        if df_column.dtype.kind in {'f', 'i'}:  # Numeric columns use np.nan.
            return df_column.fillna(np.nan).to_numpy()
        else:  # For non-numeric columns, return NaN string.
            return df_column.fillna("NaN").to_numpy()

    def _save_cutouts(
        self, save_dir, flux_cutouts, ivar_cutouts, mask_cutouts, cutouts_df
    ):
        """
        Save the cutouts to disk.

        Args:
            save_dir (string): where .h5 files are stored
            flux_cutouts (np.array, size=(n_galaxies,npix,npix)): Flux of each
                cutout.
            ivar_cutouts (np.array, size=(n_galaxies,npix,npix)): Inverse
                variance of each cutout.
            mask_cutouts (np.array, size=(n_galaxies,npix,npix)): Boolean masks
            cutouts_df (pd.dataframe): assumed to contain 'RA' 'DEC'
        """

        # track original list of unique indices
        cutouts_idx_orig = cutouts_df.index.to_numpy()

        # group by healpix coordinate
        cutouts_df = cutouts_df.groupby("healpix")

        # write to the file for the corresponding healpix coordinate
        for group in cutouts_df:

            group_filename = (save_dir + f"healpix={group[0]}/001-of-001.hdf5")
            group_cutouts_df = group[1]

            # only index flux cutouts and ivar cutouts for members of this group
            cutouts_idx_group = group_cutouts_df.index.to_numpy()
            idxs_cutouts = np.asarray(
                np.where(np.isin(cutouts_idx_orig, cutouts_idx_group))[0]
            )
            group_flux_cutouts = np.asarray(flux_cutouts)[idxs_cutouts]
            group_ivar_cutouts = np.asarray(ivar_cutouts)[idxs_cutouts]
            group_mask_cutouts = np.asarray(mask_cutouts)[idxs_cutouts]

            # Create the output directory if it does not exist
            out_path = os.path.dirname(group_filename)
            if not os.path.exists(out_path):
                print("Creating output directory: ", out_path)
                os.makedirs(out_path, exist_ok=True)

            if os.path.exists(group_filename):
                # Load the existing file and concatenate the data with current
                # data
                with h5py.File(group_filename, 'a') as hdf5_file:

                    # Check if we already wrote these object_ids earlier
                    # (to accomodate user running the script more than once)
                    h5_idx_already_written = np.isin(
                        hdf5_file['object_id'][:], cutouts_idx_group
                    )
                    group_idx_already_written = np.isin(
                        cutouts_idx_group, hdf5_file['object_id'][:]
                    )
                    group_idx_to_write = ~group_idx_already_written

                    # convert to numpy indexing, which is better for h5py
                    h5_idx_already_written = np.where(h5_idx_already_written)[0]
                    group_idx_already_written = np.where(
                        group_idx_already_written
                    )[0]
                    group_idx_to_write = np.where(group_idx_to_write)[0]

                    # over-write indices that are already written
                    if len(h5_idx_already_written) > 0:

                        hdf5_file['image_flux'][h5_idx_already_written] = (
                            group_flux_cutouts[group_idx_already_written]
                        )
                        hdf5_file['image_ivar'][h5_idx_already_written] = (
                            group_ivar_cutouts[group_idx_already_written]
                        )
                        hdf5_file['image_mask'][h5_idx_already_written] = (
                            group_mask_cutouts[group_idx_already_written]
                        )
                        hdf5_file['object_id'][h5_idx_already_written] = (
                            cutouts_idx_group[group_idx_already_written]
                        )
                        # Append all info from dataframe (including 'RA','DEC',
                        # & 'healpix')
                        for key in group_cutouts_df:
                            # If this key does not already exist, we skip it
                            if key not in hdf5_file:
                                continue
                            # use .iloc here to index by row (dataframe index is
                            # not consistent with rows number)
                            hdf5_file[key][h5_idx_already_written] = (
                                self._extract_numpy(
                                    group_cutouts_df.iloc[
                                        group_idx_already_written
                                    ][key]
                                )
                            )

                    # create new rows for new indices
                    if len(h5_idx_already_written) < len(cutouts_idx_group):

                        # Append image data
                        shape = group_flux_cutouts[group_idx_to_write].shape
                        hdf5_file['image_flux'].resize(
                            hdf5_file['image_flux'].shape[0] + shape[0], axis=0
                        )
                        hdf5_file['image_flux'][-shape[0]:] = (
                            group_flux_cutouts[group_idx_to_write]
                        )

                        # Append ivar data
                        hdf5_file['image_ivar'].resize(
                            hdf5_file['image_ivar'].shape[0] + shape[0], axis=0
                        )
                        hdf5_file['image_ivar'][-shape[0]:] = (
                            group_ivar_cutouts[group_idx_to_write]
                        )

                        # Append image masks
                        hdf5_file['image_mask'].resize(
                            hdf5_file['image_mask'].shape[0] + shape[0], axis=0
                        )
                        hdf5_file['image_mask'][-shape[0]:] = (
                            group_mask_cutouts[group_idx_to_write]
                        )

                        # Append unique image ID
                        hdf5_file['object_id'].resize(
                            hdf5_file['object_id'].shape[0] + shape[0], axis=0
                        )
                        hdf5_file['object_id'][-shape[0]:] = (
                            cutouts_idx_group[group_idx_to_write]
                        )


                        # Append all info from dataframe (including 'RA','DEC',
                        # & 'healpix')
                        for key in group_cutouts_df:
                            # If this key does not already exist, we skip it
                            if key not in hdf5_file:
                                continue
                            shape = group_cutouts_df.iloc[
                                group_idx_to_write
                            ][key].shape
                            hdf5_file[key].resize(
                                hdf5_file[key].shape[0] + shape[0], axis=0
                            )
                            hdf5_file[key][-shape[0]:] = self._extract_numpy(
                                group_cutouts_df.iloc[group_idx_to_write][key]
                            )

            else:
                # This is the first time we write the file, so we define the
                # datasets
                with h5py.File(group_filename, 'w') as h5f:
                    # save the image data.
                    shape = group_flux_cutouts.shape
                    h5f.create_dataset(
                        'image_flux', data=group_flux_cutouts,
                        compression="gzip", chunks=True,
                        maxshape=(None, *shape[1:])
                    )
                    h5f['image_flux'].attrs['description'] = (
                        'Flux values of the cutout images.'
                    )

                    shape = group_ivar_cutouts.shape
                    h5f.create_dataset(
                        'image_ivar', data=group_ivar_cutouts,
                        compression="gzip", chunks=True,
                        maxshape=(None,*shape[1:])
                    )
                    h5f['image_ivar'].attrs['description'] = (
                        '''
                        Inverse variance maps of the cutout images. Accounts
                        for background noise (sky brightness & read noise)'
                        '''
                    )

                    shape = group_mask_cutouts.shape
                    h5f.create_dataset(
                        'image_mask', data=group_mask_cutouts,
                        compression="gzip", chunks=True,
                        maxshape=(None,*shape[1:])
                    )
                    h5f['image_mask'].attrs['description'] = (
                        'Image mask for nans and zero exposure time.'
                    )

                    # save a unique object_id for each object, tied to their index in
                    # the DataFrame.
                    h5f.create_dataset(
                        'object_id', data=group_cutouts_df.index.to_numpy(),
                        compression="gzip", chunks=True, maxshape=(None,)
                    )

                    # Add the remaining data.
                    descriptions={
                        'RA':'Right Ascension of cutout center.',
                        'DEC':'Declination of cutout center.',
                        'healpix':'healpix pixel of cutout.'
                    }
                    for key in group_cutouts_df:
                        h5f.create_dataset(
                                key, data=self._extract_numpy(
                                    group_cutouts_df.loc[:, key]
                                ),
                                compression="gzip",
                                chunks=True, maxshape=(None,)
                            )
                        if key in descriptions.keys():
                            h5f[key].attrs['description'] = descriptions[key]

    def _download_tile(self, target_name, proposal_id):
        """
        Args:
            target_name (string): ex: 'COSMOS16-10'
            proposal_id (int): proposal id for target in mast query.

        Returns:
            (str): path the file was downloaded to.
        """

        # for each tile, make a folder. In that folder, download the calacs
        # and save cutouts
        target_dir = os.path.join(self.downloads_folder, target_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # query MAST using astroquery
        obs_table = Observations.query_criteria(
            proposal_id=proposal_id, target_name=target_name,
            obs_collection='HST', provenance_name='CALACS',
            filters=BAND, dataproduct_type='image'
        )

        data_idx=0
        # if multiple data products found for the same tile,
        if len(obs_table) > 1:
            # use the data product with longer exposure time
            data_idx = np.argmax(obs_table['t_exptime'])
        product_uri = obs_table[data_idx]["dataURL"]

        # download calacs .fits from MAST
        local_path = os.path.join(target_dir, 'calacs.fits')
        _ = Observations.download_file(
            uri=product_uri,
            local_path=local_path
        )

        return local_path

    def process_tiles(self, tile_name_list, proposal_id_list):
        """
        Args:
            tile_name_list ([string]): ex: 'COSMOS16-10'
            proposal_id_list ([int]): proposal id associated to each tile in
                tile_name_list (for COSMOS, either 9822 or 10092)

        Returns:
            - downloads
            saves 'galaxies.h5' and 'foregrounds.h5' to the folder
        """

        # download tiles (if already downloaded, astroquery will skip)
        assert len(tile_name_list) == len(proposal_id_list)
        with ThreadPoolExecutor(max_workers=5) as executor:
            fits_file_list = list(
                executor.map(
                    self._download_tile, tile_name_list, proposal_id_list
                )
            )

        # assign each galaxy to a .fits file (modifies galaxy_df in place)
        fits_col_name = 'fits_file'
        cutout_utils.get_file_for_targets(
            self.cosmos_df, fits_file_list, col_name=fits_col_name
        )

        # Report fraction of galaxies that are assigned a fits file.
        hit_rate = (
            self.cosmos_df['fits_file'].notna().sum() / len(self.cosmos_df)
        )
        print(f'Fraction of galaxies with assigned fits files: {hit_rate:.3f}')

        # generate our random catalog from the boundaries of each of our fits
        # files.
        randoms_df = cutout_utils.get_randoms_from_files(
            self.cosmos_df, fits_file_list, NUMPIX, col_name=fits_col_name
        )

        # add in healpix information
        self.cosmos_df['healpix'] = hp.ang2pix(
                nside=HEALPIX_NSIDE, theta=self.cosmos_df['RA'].to_numpy(),
                phi=self.cosmos_df['DEC'].to_numpy(),
                lonlat=True, nest=True
            )
        randoms_df['healpix'] = hp.ang2pix(
                nside=HEALPIX_NSIDE, theta=randoms_df['RA'].to_numpy(),
                phi=randoms_df['DEC'].to_numpy(),
                lonlat=True, nest=True
            )
        # save filter band
        self.cosmos_df['image_band'] = np.repeat([BAND],len(self.cosmos_df))
        randoms_df['image_band'] = np.repeat([BAND],len(randoms_df))

        # now each galaxy has a .fits file assigned. For each .fits file, make
        # cutouts, and save as an hdf5.
        for fits_file in tqdm(fits_file_list):

            # load up image, file and wcs
            with fits.open(fits_file) as hdu:
                data_header = hdu[1].header
                flux_tile = hdu[1].data
                weight_tile = hdu[2].data
                wcs = WCS(data_header,hdu)

            # Save galaxies
            # index sources that match this file
            tile_galaxies = self.cosmos_df[
                self.cosmos_df['fits_file']==fits_file
            ]

            # make image and wht cutouts
            flux_cutouts, ivar_cutouts, mask_cutouts, idxs_cutouts = (
                self.make_cutouts(tile_galaxies, flux_tile, weight_tile, wcs)
            )
            tile_galaxies = tile_galaxies.loc[idxs_cutouts]

            self._save_cutouts(
                os.path.join(self.output_dir, 'galaxies/'), flux_cutouts,
                ivar_cutouts, mask_cutouts, tile_galaxies
            )

            # Save randoms.
            # index sources that match this file
            tile_randoms = randoms_df[
                randoms_df['fits_file']==fits_file
            ]

            # make image and wht cutouts
            r_flux_cutouts, r_ivar_cutouts, r_mask_cutouts, r_idxs_cutouts = (
                self.make_cutouts(tile_randoms, flux_tile, weight_tile, wcs)
            )
            tile_randoms = tile_randoms.loc[r_idxs_cutouts]

            self._save_cutouts(
                os.path.join(self.output_dir, 'randoms/'), r_flux_cutouts,
                r_ivar_cutouts, r_mask_cutouts, tile_randoms
            )


##############
#### Main ####
##############
def main(args):
    """ Main function for script. """
    # make the cutout class.
    my_cosmos_cutouts = CosmosCutouts(
        metadata_main_path=args.metadata_path,
        morphology_path=args.morphology_path,
        downloads_folder=args.downloads_folder,
        output_dir=args.output_dir,
        nan_tolerance=args.nan_tolerance,
        zero_tolerance=args.zero_tolerance
    )

    target_list = fill_target_lists()
    target_names = np.concatenate(list(target_list.values()))
    pids = np.concatenate(
        [np.full(target_list[key].shape, int(key)) for key in target_list]
    )

    print('num tiles: ', len(target_names))
    if args.debug:
        print('Debugging with first 10 tiles')
        target_names = target_names[:10]
        pids = pids[:10]

    # test with the first tile
    my_cosmos_cutouts.process_tiles(target_names, pids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Downloads HST COSMOS tiles from MAST"
    )
    parser.add_argument(
        "--nan_tolerance",
        type=float,
        default=1.0,
        help=("Fraction of values that can be nan before cutout is ignored. "+
            "If 1.0, all values can be nans.")
    )
    parser.add_argument(
        "--zero_tolerance",
        type=float,
        default=1.0,
        help=("fraction of values that can be zeros before cutout is ignored. "+
            "If 1.0, all values can be zeros.")
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="metadata path for COSMOS galaxies."
    )
    parser.add_argument(
        "--morphology_path",
        type=str,
        default=None,
        help="morphology path for COSMOS galaxies."
    )
    parser.add_argument(
        "--downloads_folder",
        type=str,
        default=None,
        help="Where to store COSMOS downloads from MAST."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store cutouts."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run on first 10 tiles"
    )

    args = parser.parse_args()
    main(args)
