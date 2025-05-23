"""Modified Hugging Face loader based on implementation in MMU."""
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datasets
import h5py
import numpy as np
from datasets import Array2D, Features, Value
from datasets.data_files import DataFilesPatternsDict

from build_parent_sample import NUMPIX

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = r"""% CITATION

# COSMOS Survey Paper
@article{Koekemoer_2007,
   title={The COSMOS Survey:
                    Hubble Space Telescope
                    Advanced Camera for Surveys Observations and Data Processing},
   volume={172},
   ISSN={1538-4365},
   url={http://dx.doi.org/10.1086/520086},
   DOI={10.1086/520086},
   number={1},
   journal={The Astrophysical Journal Supplement Series},
   publisher={American Astronomical Society},
   author={Koekemoer, A. M. and Aussel, H. and Calzetti, D. and Capak, P. and Giavalisco, M. and Kneib, J.‐P. and Leauthaud, A. and Le Fevre, O. and McCracken, H. J. and Massey, R. and Mobasher, B. and Rhodes, J. and Scoville, N. and Shopbell, P. L.},
   year={2007},
   month=sep, pages={196–202} }

# Galaxy Zoo Hubble
@article{Willett_2016,
   title={Galaxy Zoo: morphological classifications for 120 000 galaxies inHSTlegacy imaging},
   volume={464},
   ISSN={1365-2966},
   url={http://dx.doi.org/10.1093/mnras/stw2568},
   DOI={10.1093/mnras/stw2568},
   number={4},
   journal={Monthly Notices of the Royal Astronomical Society},
   publisher={Oxford University Press (OUP)},
   author={Willett, Kyle W. and Galloway, Melanie A. and Bamford, Steven P. and Lintott, Chris J. and Masters, Karen L. and Scarlata, Claudia and Simmons, B. D. and Beck, Melanie and Cardamone, Carolin N. and Cheung, Edmond and Edmondson, Edward M. and Fortson, Lucy F. and Griffith, Roger L. and Häußler, Boris and Han, Anna and Hart, Ross and Melvin, Thomas and Parrish, Michael and Schawinski, Kevin and Smethurst, R. J. and Smith, Arfon M.},
   year={2016},
   month=oct, pages={4176–4203} }

"""

_DESCRIPTION = """\
Image dataset based on HST COSMOS
"""

DEFAULT_BANDS = ('f814w',)


class CustomBuilderConfig(datasets.BuilderConfig):
    """Based BuilderConfig."""
    def __init__(
        self, float_features=None, image_size=NUMPIX, bands=DEFAULT_BANDS,
        **kwargs
    ):
        """Custom builder config for HST Cosmos dataset.

        Args:
            image_size: The size of the images.
            bands: A list of bands for the dataset.
            **kwargs: Keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)
        self.image_size = image_size
        self.bands = bands
        self.float_features = float_features or []

HST_FLOATS = [
    'FLUX_BEST_HI', 'FLUX_RADIUS_HI', 'MAG_BEST_HI', 'KRON_RADIUS_HI'
]

class HstCOSMOS(datasets.GeneratorBasedBuilder):
    """HST COSMOS Dataset."""

    BUILDER_CONFIGS = [
        CustomBuilderConfig(
            name="hst-cosmos-galaxies",
            data_files=DataFilesPatternsDict.from_patterns(
                {"train": [r"COSMOS/galaxies/healpix=*/*.hdf5"]}
            ),
            description="HST-COSMOS galaxies",
            float_features=HST_FLOATS,
        ),
        CustomBuilderConfig(
            name="hst-cosmos-randoms",
            data_files=DataFilesPatternsDict.from_patterns(
                {"train": [r"COSMOS/randoms/healpix=*/*.hdf5"]}
            ),
            description="HST-COSMOS randoms",
            float_features=[],
        ),
        CustomBuilderConfig(
            name="hst-cosmos-galaxies-debug",
            data_files=DataFilesPatternsDict.from_patterns(
                {"train": [r"COSMOS/galaxies/healpix=109025/*.hdf5"]}
            ),
            description="HST-COSMOS randoms",
            float_features=HST_FLOATS,
        ),
        CustomBuilderConfig(
            name="hst-cosmos-randoms-debug",
            data_files=DataFilesPatternsDict.from_patterns(
                {"train": [r"COSMOS/randoms/healpix=108983/*.hdf5"]}
            ),
            description="HST-COSMOS randoms",
            float_features=[],
        )
    ]

    DEFAULT_CONFIG_NAME = "hst-cosmos-galaxies"

    def _info(self):
        """Defines the features available in this dataset."""

        # Starting with all features common to image datasets
        features = {
            "image_band": Value("string"),
            "image_flux": Array2D(
                shape=(self.config.image_size, self.config.image_size),
                dtype="float32"
            ),
            "image_ivar": Array2D(
                shape=(self.config.image_size, self.config.image_size),
                dtype="float32"
            ),
            "image_mask": Array2D(
                shape=(self.config.image_size, self.config.image_size),
                dtype="uint8"
            ),
            "image_scale": Value("float32"),
        }

        # Adding all values from the catalog
        for feat in self.config.float_features:
            features[feat] = Value("float32")

        features["object_id"] = Value("string")

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=Features(features)
        )

    def _split_generators(self, dl_manager):
        """Split dataset and deal with filetype diversity."""
        if not self.config.data_files:
            raise ValueError(
                "At least one data file must be specified, but got " +
                f"data_files={self.config.data_files}"
            )
        splits = []
        for split_name, files in self.config.data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(
                datasets.SplitGenerator(
                    name=split_name, gen_kwargs={"files": files}
                )
            )
        return splits

    def _generate_examples(self, files, object_ids=None):
        """Yields examples as (key, example) tuples.

        If objects_ids=None, uses all object_ids found in the file
        """
        for j, file in enumerate(files):
            print(f"Processing file: {file}")
            with h5py.File(file, "r") as data:

                # user can provide object IDs to look for in this file
                if object_ids is not None:
                    keys = object_ids[j]

                # by default: loops through all object IDs in the file
                else:
                    keys = data["object_id"][:]

                # Preparing an index for fast searching through the catalog
                sort_index = np.argsort(data["object_id"][:])
                sorted_ids = data["object_id"][:][sort_index]

                for k in keys:
                    # Extract the indices of requested ids in the catalog
                    i = sort_index[np.searchsorted(sorted_ids, k)]

                    # Check if the found object_id matches the requested one
                    if data["object_id"][i] != k:
                        print(f"Warning: Object {k} not found in this chunk. Skipping.")
                        continue

                    # Parse image data
                    example = {
                        "image_band": data["image_band"][i].decode("utf-8"),
                        "image_flux": data["image_flux"][i],
                        "image_ivar": data["image_ivar"][i],
                        "image_mask": data["image_mask"][i].astype(bool),
                        "image_scale": data["pixel_scale"][i],
                    }

                    # Add all other requested features
                    for feat in self.config.float_features:
                        try:
                            value = data[feat][i]
                            example[feat] = (
                                float(value) if np.isscalar(value) else 0.0
                            )
                        except KeyError:
                            print(
                                f"Warning: Feature '{feat}' not found in " +
                                "the dataset."
                            )
                            example[feat] = 0.0

                    # Add object_id
                    example["object_id"] = str(data["object_id"][i])

                    yield example["object_id"], example
