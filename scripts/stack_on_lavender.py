#!/usr/bin/env python
"""Script runs stack on lavender galaxies: both isolated and blend"""
import os
import subprocess
import numpy as np
from astropy.table import Table, vstack
import utils
import load_wld_data as lwd
import run_stack
import run_scarlet
# import copy
from scipy import spatial
import multiprocessing
import pickle
import time
MAIN_PATH = '/global/cscratch1/sd/sowmyak/training_data'
DATA_PATH = os.path.join(MAIN_PATH, "blending_metrics")


def get_psf_sky(path, nx=41, ny=41,
                bands=("u", "g", "r", "i", "z", "y")):
    """Returns PSF and mean sky counts"""
    psfs = np.zeros((len(bands), nx, ny))
    sky_counts = np.zeros(len(bands))
    for i, b in enumerate(bands):
        input_path = path + "/lavender/gal_pair_%s_wldeb.fits"%b
        LSST = lwd.get_wld_data(input_path)
        psfs[i] = lwd.get_psf(LSST)
        sky_counts[i] = lwd.get_sky_counts(LSST)
    return psfs, sky_counts


def load_catalogs(path):
    filename = os.path.join(path, 'lavender/gal_pair_catalog.fits')
    input_cat = Table.read(filename, format='fits')
    filename = os.path.join(path, 'lavender_blend_param.tab')
    blend_cat = Table.read(filename, format='ascii')
    return input_cat, blend_cat


def get_true_peaks(input_cat1, input_cat2):
    peak1 = np.array([input_cat1['dx'] + 40, input_cat1['dy'] + 40]).T
    peak2 = np.array([input_cat2['dx'] + 40, input_cat2['dy'] + 40]).T
    return np.stack((peak1, peak2), axis=1)


def get_blends(path, num, i_mag=24):
    """Returns catalogs with blend information"""
    input_cat, blend_cat = load_catalogs(path)
    tot_num = len(blend_cat)
    # Pick pairs only from training set where atleast one is observable
    cond1 = (input_cat['i_ab'][:tot_num] <= 25.3) & (input_cat['i_ab'][tot_num:] < 27)
    cond2 = (blend_cat['is_validation'] == 0)
    q, = np.where(cond1 & cond2)
    #choice = np.random.choice(q, size=num, replace=False)
    in_cat1 = input_cat[:tot_num][:num]
    in_cat2 = input_cat[tot_num:][:num]
    return in_cat1, in_cat2, blend_cat[:num]


def get_images(path, blend_cat):
    filename = os.path.join(path, 'lavender/stamps.pickle')
    X_train, Y_train, X_val, Y_val = utils.load_data(filename)
    blend_images = X_train['blend_image'][blend_cat['nn_id']]
    m, s = blend_cat['mean'], blend_cat['std']
    blend_images = blend_images.T * s + m
    true1 = Y_train['Y1'][blend_cat['nn_id']].T * s + m
    true2 = Y_train['Y2'][blend_cat['nn_id']].T * s + m
    return blend_images.T, true1.T, true2.T


def get_data(path, num, bands=('u', 'g', 'r', 'i', 'z', 'y')):
    """Loadlavender images of blend and isolated galaxies and saves to a dict
    Keyword Arguments
        path -- path to file with images
        num --  Number of blends for analysis
    Returns
        data -- dict with data for analysis of num blends
    """
    psfs, sky_counts = get_psf_sky(path, bands=bands)
    # in_cat has true center values for scarlet
    # blend_cat has nn_id: index of images in lavenddr
    in_cat1, in_cat2, blend_cat = get_blends(path, num)
    blend_images, true1, true2 = get_images(path, blend_cat)
    # import ipdb;ipdb.set_trace()
    peaks = get_true_peaks(in_cat1, in_cat2)
    data = {'blend': blend_images,
            'true1': true1,
            'true2': true2,
            'psfs': psfs,
            'sky_counts': sky_counts,
            'peaks': peaks,
            'blend_cat': blend_cat}
    return data


def run_stack_wld(data, num, path, i_index):
    """Runs stack on blend and isolated objects"""
    var = data['blend'][num, :, :, i_index] + data['sky_counts'][i_index]
    run_stack.get_good_childrn(data['blend'][num, :, :, i_index].astype(np.float32),
                               psf_array=data['psfs'][i_index],
                               variance_array=var.astype(np.float32),
                               output_path=path + "/blnd_stck.fits")
    var = data['true1'][num, :, :, 0] + data['sky_counts'][i_index]
    run_stack.get_good_childrn(data['true1'][num, :, :, 0].astype(np.float32),
                               psf_array=data['psfs'][i_index],
                               variance_array=var.astype(np.float32),
                               output_path=path + "/ind_1_stck.fits")
    var = data['true2'][num, :, :, 0] + data['sky_counts'][i_index]
    run_stack.get_good_childrn(data['true2'][num, :, :, 0].astype(np.float32),
                               psf_array=data['psfs'][i_index],
                               variance_array=var.astype(np.float32),
                               output_path=path + "/ind_2_stck.fits")


def scarlet_stack_true_cent(data, num, path, i_index):
    """Run scarlet with true centers"""
    # Transpose as scarlet needs in format (bands, nx, ny)
    blend, rej_tru = run_scarlet.fit(data['blend'][num].T,
                                     data['peaks'][num],
                                     data['psfs'],
                                     data['sky_counts']**0.5)
    deb_img_tru = run_scarlet.deb_images(blend, data['blend'][num].T)
    indxs = np.array([0, 1])
    deb_indx = np.delete(indxs, rej_tru)
    print ("Scarlet rejected object:", rej_tru)
    for i, indx in enumerate(deb_indx):
        fname = path + "/deb_%i_tru.fits"%indx
        var = deb_img_tru[i] + data['sky_counts'][i_index]
        run_stack.get_good_childrn(deb_img_tru[i].astype(np.float32),
                                   psf_array=data['psfs'][i_index],
                                   variance_array=var.astype(np.float32),
                                   output_path=fname)
    fname = path + "/deb_tru_images"
    with open(fname, 'wb') as handle:
        pickle.dump(deb_img_tru, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def get_stack_peaks(cat, peaks, tolerance=5):
    """Returns centers of stack detected objects on blend image"""
    z2 = np.zeros((len(cat), 2))
    z2[:, 0] = cat['base_SdssCentroid_y']
    z2[:, 1] = cat['base_SdssCentroid_x']
    indxs, cent = [], []
    z_tree = spatial.KDTree(peaks)
    match = z_tree.query(z2, distance_upper_bound=tolerance)
    indxs = match[1]
    for i in range(len(indxs)):
        if np.isnan(match[0][i]):
            indxs[i] = 2 + i
    cent = z2
    return indxs, cent


def check_detection_scarlet(data, num, i_index):
    fname = os.path.join(DATA_PATH, 'blend_{0}/blnd_stck.fits'.format(num))
    cat = Table.read(fname, hdu=1)
    count = 0
    num_diff = len(cat)
    while ((num_diff > 0) & (count < 5)):
        indxs, peaks = get_stack_peaks(cat, data['peaks'][num])
        sel = range(len(peaks))
        bl, rej_tru = run_scarlet.fit(data['blend'][num].T,
                                      peaks,
                                      data['psfs'],
                                      data['sky_counts']**0.5)
        sel = np.delete(sel, rej_tru)
        mask = [not np.isclose(np.sum(bl.get_model(i)), 0) for i in range(len(bl.sources))]
        cat = cat[sel][mask]
        diff_im = data['blend'][num, :, :, i_index] - bl.get_model()[i_index].T
        v = diff_im + data['sky_counts'][i_index]
        cat2 = run_stack.get_good_childrn(diff_im.astype(np.float32),
                                          psf_array=data['psfs'][i_index],
                                          variance_array=v.astype(np.float32),
                                          detect=True)
        num_diff = len(cat2)
        cat = vstack([cat, cat2], join_type='inner')
        count += 1
    fname = os.path.join(DATA_PATH, 'blend_{0}/dt_stck_scrlt.fits'.format(num))
    cat.write(fname, format='fits', overwrite=True)
    return cat


def scarlet_stack_stck_cent(data, num, path, i_index):
    """Run scarlet with centers from dm stack"""
    # Transpose as scarlet needs in format (bands, nx, ny)
    cat = check_detection_scarlet(data, num, i_index)
    indxs, peaks = get_stack_peaks(cat, data['peaks'][num])
    blend, rej_tru = run_scarlet.fit(data['blend'][num].T,
                                     peaks,
                                     data['psfs'],
                                     data['sky_counts']**0.5)
    deb_img_stck = run_scarlet.deb_images(blend, data['blend'][num].T)
    deb_indx = np.delete(indxs, rej_tru)
    print ("Scarlet rejected object:", rej_tru)
    for i, indx in enumerate(deb_indx):
        fname = path + "/deb_%i_stck.fits"%indx
        var = deb_img_stck[i] + data['sky_counts'][i_index]
        run_stack.get_good_childrn(deb_img_stck[i].astype(np.float32),
                                   psf_array=data['psfs'][i_index],
                                   variance_array=var.astype(np.float32),
                                   output_path=fname)
    fname = path + "/deb_stck_images"
    with open(fname, 'wb') as handle:
        pickle.dump(deb_img_stck, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def sort_by_dist(point, xs, ys):
    dist = np.hypot(xs - point[0], ys - point[1])
    arr = np.array([range(len(xs)), dist])
    new_arr = np.sort(arr, axis=1)
    return int(new_arr[0])


def run_analysis(data, num, path, i_index):
    print ("running Analysis", num)
    ind_path = os.path.join(path, "blend_" + str(num))
    if os.path.isdir(ind_path) is False:
        subprocess.call(["mkdir", ind_path])
    run_stack_wld(data, num, ind_path, i_index)
    scarlet_stack_true_cent(data, num, ind_path, i_index)
    scarlet_stack_stck_cent(data, num, ind_path, i_index)


def run_batch(data, num, i_index):
    print("running batch", num / 30)
    pool = multiprocessing.Pool(30)
    for i in range(num, num + 30):
        pool.apply_async(run_analysis, [data, i, DATA_PATH, i_index])
        # run_analysis(data, i, DATA_PATH)
    pool.close()
    pool.join()


def main(Args):
    bands = ('g', 'r', 'i')
    i_index = 2
    start = time.time()
    print ("starting at ", time.time())
    np.random.seed(0)
    data = get_data(MAIN_PATH, Args.num, bands)
    for i in range(0, int(Args.num / 30) * 30, 30):
        run_analysis(data, i, DATA_PATH, i_index)
        #run_batch(data, i, i_index)
        time.sleep(45)
    filename = os.path.join(DATA_PATH, "data.pickle")
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print("time taken: ", end - start)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', default=49000, type=int,
                        help="# of distinct galaxy pairs [Default:100]")
    args = parser.parse_args()
    main(args)
