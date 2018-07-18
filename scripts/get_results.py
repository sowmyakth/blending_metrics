import os
import numpy as np
from astropy.table import Table
import pickle
import galsim
import stack_on_lavender as sol

MAIN_PATH = '/global/cscratch1/sd/sowmyak/training_data'
DATA_PATH = os.path.join(MAIN_PATH, "blending_metrics")


def get_true_data():
    filename = os.path.join(DATA_PATH, "data.pickle")
    with open(filename, 'rb') as handle:
            data = pickle.load(handle)
    return data


def get_normalized_dist(cat):
    num = cat['distance_neighbor']
    d1 = np.hypot(cat['sigma_m'], cat['psf_sigm'])
    d2 = np.hypot(cat['sigma_neighbor'], cat['psf_sigm'])
    denom = d1 + d2
    return num / denom


def get_det_perc(indx):
    cat = Table.read(os.path.join(DATA_PATH,
                                  'blend_{0}/blnd_stck.fits'.format(indx)),
                     hdu=1)
    num = len(cat)
    cat1 = Table.read(os.path.join(DATA_PATH,
                                   'blend_{0}/ind_1_stck.fits'.format(indx)),
                      hdu=1)
    cat2 = Table.read(os.path.join(DATA_PATH,
                                   'blend_{0}/ind_2_stck.fits'.format(indx)),
                      hdu=1)
    denom = len(cat1) + len(cat2)
    if num > 2:
        clss = 'shred'
    elif num == 0:
        clss = 'undet'
    elif num == denom:
        if num == 2:
            clss = 'recog_blnd'
        elif num == 1:
            clss = 'undet_blnd'
    elif (denom == 2) & (num == 1):
        fname = os.path.join(DATA_PATH,
                             'blend_{0}/dt_stck_scrlt.fits'.format(indx))
        if os.path.isfile(fname):
            cat_det = Table.read(fname, hdu=1)
            if len(cat_det) > 2:
                clss = 'iter_shred'
            elif (len(cat_det) == 2):
                clss = 'iter_recog'
            else:
                clss = 'unrecog_blnd'
        else:
            clss = 'check'
    else:
        clss = 'check'
    if denom == 0:
        perc = -1
    else:
        perc = num / denom
    return clss, perc


def get_entry(fname):
    try:
        cat = Table.read(os.path.join(DATA_PATH, fname), hdu=1)
        if len(cat) == 1:
            x = float(cat['ext_shapeHSM_HsmSourceMoments_y'][0])
            y = float(cat['ext_shapeHSM_HsmSourceMoments_x'][0])
            sigma = float(cat['ext_shapeHSM_HsmShapeRegauss_sigma'][0])
            flux = float(cat['base_SdssShape_flux'][0])
            e1 = float(cat['ext_shapeHSM_HsmShapeRegauss_e1'][0])
            e2 = float(cat['ext_shapeHSM_HsmShapeRegauss_e2'][0])
            return x, y, sigma, flux, e1, e2
        else:
            return -1, -1, -1, -1, -1, -1
    except FileNotFoundError:
        return -1, -1, -1, -1, -1, -1


def print_classes(result, names):
    tot = 0
    for i, name in enumerate(names):
        print(name, ": ", len(result[name]['blnd_indx']))
        tot += len(result[name]['blnd_indx'])
    print ("total", tot)


def results_all(data):
    names = ('undet_blnd', 'unrecog_blnd', 'recog_blnd', 'iter_recog', 'undet', 'shred', 'check', 'iter_shred')
    vals = np.array(['blnd_indx', 'det_perc', 'unit_dist', 'distance', 'min_purity'])
    vals = np.append(vals, ['gal1_x_tru_scrlt', 'gal1_y_tru_scrlt', 'gal2_x_tru_scrlt', 'gal2_y_tru_scrlt'])
    vals = np.append(vals, ['gal1_e1_tru_scrlt', 'gal1_e2_tru_scrlt', 'gal2_e1_tru_scrlt', 'gal2_e2_tru_scrlt'])
    vals = np.append(vals, ['gal1_flux_tru_scrlt', 'gal1_sigma_tru_scrlt', 'gal2_flux_tru_scrlt', 'gal2_sigma_tru_scrlt'])
    vals = np.append(vals, ['gal1_x_stck_scrlt', 'gal1_y_stck_scrlt', 'gal2_x_stck_scrlt', 'gal2_y_stck_scrlt'])
    vals = np.append(vals, ['gal1_e1_stck_scrlt', 'gal1_e2_stck_scrlt', 'gal2_e1_stck_scrlt', 'gal2_e2_stck_scrlt'])
    vals = np.append(vals, ['gal1_flux_stck_scrlt', 'gal1_sigma_stck_scrlt', 'gal2_flux_stck_scrlt', 'gal2_sigma_stck_scrlt'])
    vals = np.append(vals, ['gal1_x_tru', 'gal1_y_tru', 'gal2_x_tru', 'gal2_y_tru'])
    vals = np.append(vals, ['gal1_e1_tru', 'gal1_e2_tru', 'gal2_e1_tru', 'gal2_e2_tru'])
    vals = np.append(vals, ['gal1_x_stck', 'gal1_y_stck', 'gal2_x_stck', 'gal2_y_stck'])
    vals = np.append(vals, ['gal1_e1_stck', 'gal1_e2_stck', 'gal2_e1_stck', 'gal2_e2_stck'])
    vals = np.append(vals, ['gal1_flux_stck', 'gal1_sigma_stck', 'gal2_flux_stck', 'gal2_sigma_stck'])
    vals = np.append(vals, ['gal1_x_ind_stck', 'gal1_y_ind_stck', 'gal2_x_ind_stck', 'gal2_y_ind_stck'])
    vals = np.append(vals, ['gal1_e1_ind_stck', 'gal1_e2_ind_stck', 'gal2_e1_ind_stck', 'gal2_e2_ind_stck'])
    vals = np.append(vals, ['gal1_flux_ind_stck', 'gal1_sigma_ind_stck', 'gal2_flux_ind_stck', 'gal2_sigma_ind_stck'])
    result = {}
    for name in names:
        result[name] = {}
        for val in vals:
            result[name][val] = []
    unit_dist = get_normalized_dist(data['blend_cat'])
    for i in range(len(data['blend_cat'])):
        clss, dp = get_det_perc(i)
        result[clss]['blnd_indx'].append(i)
        result[clss]['det_perc'].append(dp)
        result[clss]['unit_dist'].append(unit_dist[i])
        result[clss]['distance'].append(data['blend_cat']['distance_neighbor'][i])
        min_purity = np.min([data['blend_cat']['purity'][i],
                            data['blend_cat']['purity_neighbor'][i]])
        result[clss]['min_purity'].append(min_purity)
        # dm output
        fname = 'blend_{0}/blnd_stck.fits'.format(i)
        cat = Table.read(os.path.join(DATA_PATH, fname), hdu=1)
        indxs, cent = sol.get_stack_peaks(cat, data['peaks'][i])
        q1, = np.where(indxs == 0)
        if len(q1) > 1:
            print("error", i)
        elif len(q1) == 0:
            result[clss]['gal1_x_stck'].append(-1)
            result[clss]['gal1_y_stck'].append(-1)
            result[clss]['gal1_flux_stck'].append(-1)
            result[clss]['gal1_sigma_stck'].append(-1)
            result[clss]['gal1_e1_stck'].append(-1)
            result[clss]['gal1_e2_stck'].append(-1)
        else:
            ind1 = q1[0]
            result[clss]['gal1_x_stck'].append(cent[ind1][1])
            result[clss]['gal1_y_stck'].append(cent[ind1][0])
            result[clss]['gal1_flux_stck'].append(float(cat['base_SdssShape_flux'][ind1]))
            result[clss]['gal1_sigma_stck'].append(float(cat['ext_shapeHSM_HsmShapeRegauss_sigma'][ind1]))
            result[clss]['gal1_e1_stck'].append(float(cat['ext_shapeHSM_HsmShapeRegauss_e1'][ind1]))
            result[clss]['gal1_e2_stck'].append(float(cat['ext_shapeHSM_HsmShapeRegauss_e2'][ind1]))
        q2, = np.where(indxs == 1)
        if len(q2) > 1:
            print("error", i)
        elif len(q2) == 0:
            result[clss]['gal2_x_stck'].append(-1)
            result[clss]['gal2_y_stck'].append(-1)
            result[clss]['gal2_flux_stck'].append(-1)
            result[clss]['gal2_sigma_stck'].append(-1)
            result[clss]['gal2_e1_stck'].append(-1)
            result[clss]['gal2_e2_stck'].append(-1)
        else:
            ind2 = q2[0]
            result[clss]['gal2_x_stck'].append(cent[ind2][1])
            result[clss]['gal2_y_stck'].append(cent[ind2][1])
            result[clss]['gal2_flux_stck'].append(float(cat['base_SdssShape_flux'][ind2]))
            result[clss]['gal2_sigma_stck'].append(float(cat['ext_shapeHSM_HsmShapeRegauss_sigma'][ind2]))
            result[clss]['gal2_e1_stck'].append(float(cat['ext_shapeHSM_HsmShapeRegauss_e1'][ind2]))
            result[clss]['gal2_e2_stck'].append(float(cat['ext_shapeHSM_HsmShapeRegauss_e2'][ind2]))

        # dm output isolated
        fname1 = 'blend_{0}/ind_1_stck.fits'.format(i)
        cat1 = Table.read(os.path.join(DATA_PATH, fname1), hdu=1)
        fname2 = 'blend_{0}/ind_2_stck.fits'.format(i)
        cat2 = Table.read(os.path.join(DATA_PATH, fname2), hdu=1)
        if len(cat1) != 1:
            result[clss]['gal1_x_ind_stck'].append(-1)
            result[clss]['gal1_y_ind_stck'].append(-1)
            result[clss]['gal1_flux_ind_stck'].append(-1)
            result[clss]['gal1_sigma_ind_stck'].append(-1)
            result[clss]['gal1_e1_ind_stck'].append(-1)
            result[clss]['gal1_e2_ind_stck'].append(-1)
        else:
            ind1 = 0
            result[clss]['gal1_x_ind_stck'].append(float(cat1['ext_shapeHSM_HsmSourceMoments_y'][ind1]))
            result[clss]['gal1_y_ind_stck'].append(float(cat1['ext_shapeHSM_HsmSourceMoments_x'][ind1]))
            result[clss]['gal1_flux_ind_stck'].append(float(cat1['base_SdssShape_flux'][ind1]))
            result[clss]['gal1_sigma_ind_stck'].append(float(cat1['ext_shapeHSM_HsmShapeRegauss_sigma'][ind1]))
            result[clss]['gal1_e1_ind_stck'].append(float(cat1['ext_shapeHSM_HsmShapeRegauss_e1'][ind1]))
            result[clss]['gal1_e2_ind_stck'].append(float(cat1['ext_shapeHSM_HsmShapeRegauss_e2'][ind1]))
        if len(cat2) != 1:
            result[clss]['gal2_x_ind_stck'].append(-1)
            result[clss]['gal2_y_ind_stck'].append(-1)
            result[clss]['gal2_flux_ind_stck'].append(-1)
            result[clss]['gal2_sigma_ind_stck'].append(-1)
            result[clss]['gal2_e1_ind_stck'].append(-1)
            result[clss]['gal2_e2_ind_stck'].append(-1)
        else:
            ind2 = 0
            result[clss]['gal2_x_ind_stck'].append(float(cat2['ext_shapeHSM_HsmSourceMoments_y'][ind2]))
            result[clss]['gal2_y_ind_stck'].append(float(cat2['ext_shapeHSM_HsmSourceMoments_x'][ind2]))
            result[clss]['gal2_flux_ind_stck'].append(float(cat2['base_SdssShape_flux'][ind2]))
            result[clss]['gal2_sigma_ind_stck'].append(float(cat2['ext_shapeHSM_HsmShapeRegauss_sigma'][ind2]))
            result[clss]['gal2_e1_ind_stck'].append(float(cat2['ext_shapeHSM_HsmShapeRegauss_e1'][ind2]))
            result[clss]['gal2_e2_ind_stck'].append(float(cat2['ext_shapeHSM_HsmShapeRegauss_e2'][ind2]))

        # true values
        fname = os.path.join(MAIN_PATH,
                             'lavender_blend_combnd.fits')
        comb_blend_cat = Table.read(fname, format='fits')
        g1 = galsim.Shear(g1=comb_blend_cat['e1_1'][i], g2=comb_blend_cat['e2_1'][i])
        g2 = galsim.Shear(g1=comb_blend_cat['e1_2'][i], g2=comb_blend_cat['e2_2'][i])
        result[clss]['gal1_e1_tru'].append(g1.e1)
        result[clss]['gal1_e2_tru'].append(g1.e2)
        result[clss]['gal2_e1_tru'].append(g2.e1)
        result[clss]['gal2_e2_tru'].append(g2.e2)
        result[clss]['gal1_x_tru'].append(data['peaks'][i][0][0])
        result[clss]['gal1_y_tru'].append(data['peaks'][i][0][1])
        result[clss]['gal2_x_tru'].append(data['peaks'][i][1][0])
        result[clss]['gal2_y_tru'].append(data['peaks'][i][1][1])
        # true center -> scrlt -> stck
        x, y, sigma, flux, e1, e2 = get_entry('blend_{0}/deb_0_tru.fits'.format(i))
        result[clss]['gal1_x_tru_scrlt'].append(x)
        result[clss]['gal1_y_tru_scrlt'].append(y)
        result[clss]['gal1_flux_tru_scrlt'].append(flux)
        result[clss]['gal1_sigma_tru_scrlt'].append(sigma)
        result[clss]['gal1_e1_tru_scrlt'].append(e1)
        result[clss]['gal1_e2_tru_scrlt'].append(e2)
        x, y, sigma, flux, e1, e2 = get_entry('blend_{0}/deb_1_tru.fits'.format(i))
        result[clss]['gal2_x_tru_scrlt'].append(x)
        result[clss]['gal2_y_tru_scrlt'].append(y)
        result[clss]['gal2_flux_tru_scrlt'].append(flux)
        result[clss]['gal2_sigma_tru_scrlt'].append(sigma)
        result[clss]['gal2_e1_tru_scrlt'].append(e1)
        result[clss]['gal2_e2_tru_scrlt'].append(e2)
      # stack center -> scrlt -> stck
        x, y, sigma, flux, e1, e2 = get_entry('blend_{0}/deb_0_stck.fits'.format(i))
        result[clss]['gal1_x_stck_scrlt'].append(x)
        result[clss]['gal1_y_stck_scrlt'].append(y)
        result[clss]['gal1_flux_stck_scrlt'].append(flux)
        result[clss]['gal1_sigma_stck_scrlt'].append(sigma)
        result[clss]['gal1_e1_stck_scrlt'].append(e1)
        result[clss]['gal1_e2_stck_scrlt'].append(e2)
        x, y, sigma, flux, e1, e2 = get_entry('blend_{0}/deb_1_stck.fits'.format(i))
        result[clss]['gal2_x_stck_scrlt'].append(x)
        result[clss]['gal2_y_stck_scrlt'].append(y)
        result[clss]['gal2_flux_stck_scrlt'].append(flux)
        result[clss]['gal2_sigma_stck_scrlt'].append(sigma)
        result[clss]['gal2_e1_stck_scrlt'].append(e1)
        result[clss]['gal2_e2_stck_scrlt'].append(e2)

    fname = os.path.join(DATA_PATH, 'results_stck_scrlt_lav.pickle')
    with open(fname, 'wb') as handle:
        pickle.dump(result, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print_classes(result, names)


def main():
    #data = get_true_data()
    num = 2800
    data = sol.get_data(MAIN_PATH, num)
    results_all(data)


if __name__ == '__main__':
    main()
