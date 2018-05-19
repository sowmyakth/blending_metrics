"""Runns stack on input WLDEB fits image"""
import lsst.afw.table
import lsst.afw.image
import lsst.afw.math
import lsst.meas.algorithms
import lsst.meas.base
import lsst.meas.deblender
import lsst.meas.extensions.shapeHSM


def process(image_array, variance_array, psf_array,
            min_pix=1, bkg_bin_size=32, thr_value=5):
    """
    Function to setup the DM stack
    Args:
    -----
    hsm : bool, If True it activates the HSM shape measurement algorithm
    from the stack. This algorithm is not included by default so make sure
    that is installed before activating (default=False).
    min_pix: Minimum size in pixels of a source to be considered by the
    stack (default=1).
    bkg_bin_size: Binning of the local background in pixels (default=32).
    thr_value: SNR threshold for the detected sources to be included in the
    final catalog(default=5).
    """
    image = lsst.afw.image.ImageF(image_array)
    # Generate the variance image
    variance = lsst.afw.image.ImageF(variance_array)
    # Generate a masked image, i.e., an image+mask+variance image (with mask=None)
    masked_image = lsst.afw.image.MaskedImageF(image, None, variance)
    psf_im = lsst.afw.image.ImageD(psf_array)  # Convert to stack's format
    fkernel = lsst.afw.math.FixedKernel(psf_im)
    psf = lsst.meas.algorithms.KernelPsf(fkernel) # Create the kernel in the stack's format
    exposure = lsst.afw.image.ExposureF(masked_image) # Passing the image to the stack
    exposure.setPsf(psf) # Assign the exposure the PSF that we created
    schema = lsst.afw.table.SourceTable.makeMinimalSchema()
    config1 = lsst.meas.algorithms.SourceDetectionConfig()
    # Tweaks in the configuration that can improve detection
    # Change carefully!
    #####
    config1.tempLocalBackground.binSize = bkg_bin_size  # This changes the local background binning. The default is 32 pixels
    config1.minPixels = min_pix  # This changes the minimum size of a source. The default is 1
    config1.thresholdValue = thr_value  # This changes the detection threshold for the footprint (5 is the default)
    #####
    detect = lsst.meas.algorithms.SourceDetectionTask(schema=schema,
                                                      config=config1)
    deblend = lsst.meas.deblender.SourceDeblendTask(schema=schema)
    config1 = lsst.meas.base.SingleFrameMeasurementConfig()
    #config1.plugins.names.add('ext_shapeHSM_HsmShapeBj')
    #config1.plugins.names.add('ext_shapeHSM_HsmShapeLinear')
    #config1.plugins.names.add('ext_shapeHSM_HsmShapeKsb')
    config1.plugins.names.add('ext_shapeHSM_HsmShapeRegauss')
    config1.plugins.names.add('ext_shapeHSM_HsmSourceMoments')
    config1.plugins.names.add('ext_shapeHSM_HsmPsfMoments')
    measure = lsst.meas.base.SingleFrameMeasurementTask(schema=schema,
                                                        config=config1)
    # exposure.setWcs(wcs_in) # And assign it to the exposure
    table = lsst.afw.table.SourceTable.make(schema)  # this is really just a factory for records, not a table
    detect_result = detect.run(table, exposure)  # We run the stack (the detection task)
    catalog = detect_result.sources  # this is the actual catalog, but most of it's still empty
    deblend.run(exposure, catalog)  # run the deblending task
    measure.run(catalog, exposure)  # run the measuring task
    catalog = catalog.copy(deep=True)  # write a copy of the catalog
    return catalog  # We return a catalog object


def get_good_childrn(image_array, variance_array, psf_array, output_path=None,
                     min_pix=1, bkg_bin_size=32, thr_value=5, detect=False):
    cat = process(image_array, variance_array, psf_array, min_pix=min_pix,
                  bkg_bin_size=bkg_bin_size, thr_value=thr_value)
    mask = cat['deblend_nChild'] == 0
    # mask &= cat['base_LocalBackground_flag_badCentroid'] == False
    if detect:
        mask &= cat['base_SdssCentroid_flag'] == False
    else:
        mask &= cat['ext_shapeHSM_HsmShapeRegauss_flag'] == False
    cat_chldrn = cat[mask]
    cat_chldrn = cat_chldrn.copy(deep=True)
    if output_path:
        cat_chldrn.writeFits(output_path)
        print ("stack output saved at", output_path)
    return cat_chldrn.asAstropy()
