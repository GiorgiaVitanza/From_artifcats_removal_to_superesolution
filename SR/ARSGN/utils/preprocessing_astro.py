from astropy.io import fits
import numpy as np
from scipy.ndimage import zoom
import os

def load_and_downsample_fits(filename, downsample_factor=2, slice_index=None, output_file=None):
    """
    Load FITS file and downsample the data.
    
    - If 2D: downsample directly.
    - If 3D: select a slice, then downsample.
    - Optionally save to a new FITS file.
    
    Parameters
    ----------
    filename : str
        Path to the FITS file.
    downsample_factor : int or float
        Factor by which to reduce resolution (e.g., 2 = half size).
    slice_index : int, optional
        If 3D data, which slice to extract. If None, take the middle slice.
    output_file : str, optional
        Path to save the downsampled FITS. If None, result is only returned.
    
    Returns
    -------
    downsampled_data : np.ndarray
        2D array after slicing (if needed) and downsampling.
    """
    
    # Load data
    with fits.open(filename) as hdul:
        data = hdul[0].data.astype(float)  # ensure float for processing
        header = hdul[0].header
    
    if data is None:
        raise ValueError("No data found in FITS file.")
    
    # If 3D, take a slice
    if data.ndim == 3:
        if slice_index is None:
            slice_index = data.shape[0] // 2  # middle slice
        data_2d = data[slice_index, :, :]
    elif data.ndim == 2:
        data_2d = data
    else:
        if header['NAXIS3'] == 1 & header['NAXIS4'] == 1:       
            data_2d = np.nan_to_num(data).astype(np.float32).squeeze()
        else:
            raise ValueError(f"Unsupported data dimensions: {data.ndim}")
    
    # Compute zoom factor (reciprocal of downsample_factor)
    zoom_factor = 1.0 / downsample_factor
    downsampled_data = zoom(data_2d, zoom_factor, order=1)  # bilinear interpolation
    
    # Save to new FITS if requested
    if output_file is not None:
        hdu = fits.PrimaryHDU(downsampled_data, header=header)
        hdulist = fits.HDUList([hdu])
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        hdulist.writeto(output_file, overwrite=True)
    
    return downsampled_data
# Just get downsampled array
file_name = "image.i.EMU_1232-04A.SB75105.cont.taylor.0.restored.conv.fits"
img2d = load_and_downsample_fits("SR\ARSGN\dataset_astro\hr/"+file_name, downsample_factor=4, output_file="SR\ARSGN\dataset_astro\lr/downsampled"+file_name+".fits")

""" # Get downsampled slice and save as FITS
img3d_slice = load_and_downsample_fits(
    "data/3D_claudio_FDF_clean_tot.fits", downsample_factor=2, slice_index=10, output_file="output/downsampled_slice.fits"
)
 """