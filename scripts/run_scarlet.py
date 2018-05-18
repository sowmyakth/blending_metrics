"""Script to run scarlet deblender. Requires multiband blend image, psf and centers of object"""
import numpy as np
import scarlet
import scarlet.psf_match


def make_diff_psf(psfs, model='moffat'):
    if model == 'moffat':
        target_psf = scarlet.psf_match.fit_target_psf(psfs,
                                                      scarlet.psf_match.moffat)
    diff_kernels, psf_blend = scarlet.psf_match.build_diff_kernels(psfs,
                                                                   target_psf)
    return diff_kernels


def initialize(images, peaks,
               diff_kernels, bg_rms):
    sources = []
    rejected_sources = []
    for n, peak in enumerate(peaks):
        try:
            result = scarlet.ExtendedSource((peak[1], peak[0]), images,
                                            bg_rms, psf=diff_kernels)
            sources.append(result)
        except scarlet.source.SourceInitError:
            rejected_sources.append(n)
            print("No flux in peak {0} at {1}".format(n, peak))
    blend = scarlet.Blend(sources, images, bg_rms=bg_rms)
    return blend, rejected_sources


def fit(images, peaks, psfs, bg_rms,
        iters=200, e_rel=.015):
    diff_kernels = make_diff_psf(psfs)
    blend, rej = initialize(images, peaks,
                            diff_kernels, bg_rms)
    blend.fit(iters, e_rel=e_rel)
    blend.resize_sources()
    blend.recenter_sources()
    return blend, rej


def deb_images(blend, images):
    if len(blend.sources) == 1:
        # return [blend.get_model(m=0)[4]]
        return [images[4].T]
    else:
        im = []
        for m in range(len(blend.sources)):
            oth_indx = np.delete(range(len(blend.sources)), m)
            model_oth = np.zeros_like(images[4])
            for i in oth_indx:
                model_oth += blend.get_model(m=i)[4]
            im.append(images[4].T - model_oth.T)
    return im
