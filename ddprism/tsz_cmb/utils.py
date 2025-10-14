import numpy as np

def xyz_to_lonlat(xyz):
    """Convert Cartesian coordinates to longitude and latitude in degrees.
       For direct use in rot arg of hp.gnomview.

    Parameters
    ----------
    xyz : array-like, shape (N, 3)
        Cartesian coordinates.

    Returns
    -------
    lon : array, shape (N,)
        Longitude in degrees, in the range [0, 360).
    lat : array, shape (N,)
        Latitude in degrees, in the range [-90, 90].
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    
    ra = np.arctan2(y, x) % (2*np.pi) # wrap to [0, 2π)
    dec = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2)) # [-π/2, π/2]
    
    lon = np.degrees(ra)
    lat = np.degrees(dec)
    
    return lon, lat
