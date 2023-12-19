import numpy as np
import coord
from numba import njit
from multiprocessing import Pool
from .utils import pixel2RaDec, eval_func_tuple
from .correlation_helpers import rotate_shear, xipm_patch, M_a_patch, getAngle, Q_T
import healpy as hp
import itertools
from functools import partial
import h5py
import cupy as cp

class Correlation():
    
    """
    This is the base class for all correlation functions. It contains the methods for finding pairs and their angles, as well as loading and saving them.
    """
    
    def __init__(self, nside, phi_center, theta_center, nbins=10, theta_min=10, theta_max=170, patch_size=90, theta_Q=90, mask=None, fastmath=True):
        
        self.nside = nside
        self.nbins = nbins
        self.theta_min = theta_min/60/180*np.pi
        self.theta_max = theta_max/60/180*np.pi
        self.binedges = np.geomspace(self.theta_min, self.theta_max, self.nbins+1)
        self.bincenters = np.sqrt(self.binedges[1:] * self.binedges[:-1])*60*180/np.pi
        self.patch_size = patch_size
        self.theta_Q = theta_Q
        self.phi_center = phi_center
        self.theta_center = theta_center
        self.n_patches = len(phi_center)
        self.fastmath = fastmath
        self.M_A_patch = njit(fastmath=fastmath)(M_a_patch)
        
        if mask is not None:
            self.map_inds = np.where(mask)[0]
        else:
            self.map_inds = np.arange(hp.nside2npix(self.nside))
 

    def get_pairs_patch(self, patch_inds, ra, dec):
        all_inds, exp2phi1_temp, exp2phi2_temp = [], [], []
        cos_vartheta = np.cos(np.subtract.outer(ra, ra))*np.multiply.outer(np.cos(dec),np.cos(dec)) + np.multiply.outer(np.sin(dec),np.sin(dec))
        dist = np.arccos(np.triu(cos_vartheta, k=1))    
        
        for i in range(self.nbins):
            inds = np.where((dist > self.binedges[i]) & (dist < self.binedges[i+1]))
            ra_pairs1 = ra[inds[0]]
            dec_pairs1 = dec[inds[0]]
            ra_pairs2 = ra[inds[1]]
            dec_pairs2 = dec[inds[1]]

            all_inds.append(np.array([patch_inds[inds[0]], patch_inds[inds[1]]]))
            
            for j, _ in enumerate(ra_pairs1):
                theta1 = np.pi/2 - getAngle(ra_pairs1[j], dec_pairs1[j], ra_pairs2[j], dec_pairs2[j])
                theta2 = np.pi/2 - getAngle(ra_pairs2[j], dec_pairs2[j], ra_pairs1[j], dec_pairs1[j])
                exp2phi1 = np.cos(2*theta1) + 1j * np.sin(2*theta1)
                exp2phi2 = np.cos(2*theta2) + 1j * np.sin(2*theta2)

                exp2phi1_temp.append(exp2phi1)
                exp2phi2_temp.append(exp2phi2)
        
        exp2phi = np.array([exp2phi1_temp, exp2phi2_temp])
        return all_inds, exp2phi

    def __get_pairs_helper__(self, i):
        vec = hp.ang2vec(self.theta_center[i], self.phi_center[i])
        patch_inds = hp.query_disc(self.nside, vec=vec, radius=np.radians(self.patch_size/60))
        pix_inds = np.intersect1d(patch_inds, self.map_inds)
        ra, dec = pixel2RaDec(pix_inds, self.nside)
        inds, exp2theta, = self.get_pairs_patch(pix_inds, ra, dec)
        ninds = np.array([len(inds[i][0]) for i in range(self.nbins)])
        all_inds = np.zeros((2,ninds.sum()))
        for bin in range(self.nbins):
            all_inds[0,np.sum(ninds[:bin]):np.sum(ninds[:bin+1])] = inds[bin][0]
            all_inds[1,np.sum(ninds[:bin]):np.sum(ninds[:bin+1])] = inds[bin][1]
        
        return all_inds.astype(int), exp2theta, ninds
    
    def calculate_pairs_2PCF(self, threads=1):
        
        if threads == 1:
            pair_inds, pair_exp2phi, bins = [], [], []
            for i in range(self.n_patches):
                result = self.__get_pairs_helper__(i)
                
                pair_inds.append(result[0])
                pair_exp2phi.append(result[1])
                bins.append(result[2])

                print(f"Patch {i+1}/{self.n_patches} done.", end='\r', flush=True)
            print()
            
        else: 
            with Pool(threads) as p:
                pair_inds, pair_exp2phi, bins =  list(map(list, zip(*p.map(self.__get_pairs_helper__, range(self.n_patches)))))
            
        self.pair_inds = pair_inds
        self.pair_exp2phi = pair_exp2phi
        self.bins = bins
    
    def get_pairs_patch_M_a(self, pixels_RA_Q_patch, pixels_dec_Q_patch, Q_patch_center_RA, Q_patch_center_dec):

        cos_vartheta = np.cos(pixels_RA_Q_patch - Q_patch_center_RA)*np.cos(Q_patch_center_dec)*np.cos(pixels_dec_Q_patch) + np.sin(Q_patch_center_dec)*np.sin(pixels_dec_Q_patch)
        vartheta = np.arccos(cos_vartheta)
        sin_vartheta = np.sqrt(1-cos_vartheta**2)
        cos_phi = np.sin(pixels_RA_Q_patch - Q_patch_center_RA)*np.cos(pixels_dec_Q_patch) / sin_vartheta
        sin_phi = (np.cos(pixels_dec_Q_patch)*np.sin(Q_patch_center_dec) - np.sin(pixels_dec_Q_patch)*np.cos(Q_patch_center_dec)*np.cos(pixels_RA_Q_patch - Q_patch_center_RA)) / sin_vartheta
        cos_2phi = cos_phi*cos_phi - sin_phi*sin_phi
        sin_2phi = 2*sin_phi*cos_phi

        Q = Q_T(vartheta)

        return cos_2phi, sin_2phi, Q

    def calculate_pairs_M_a(self, threads=1):
        #TODO
        pass

    def preprocess(self, threads=1):
        
        """
        Calculates the pairs and their angles for all patches for 2PCF & aperture mass.
        """
        #TODO
        pass

    def save_pairs(self, filepath):
        with h5py.File(filepath, 'w') as fp:
            fp.attrs['nside'] = self.nside
            fp.attrs['nbins'] = self.nbins
            fp.attrs['theta_min'] = self.theta_min
            fp.attrs['theta_max'] = self.theta_max
            fp.attrs['patch_size'] = self.patch_size
            fp.attrs['theta_Q'] = self.theta_Q
            fp.attrs['n_patches'] = self.n_patches
            fp.create_dataset('map_inds', data=self.map_inds)
            fp.create_dataset('phi_center', data=self.phi_center)
            fp.create_dataset('theta_center', data=self.theta_center)
            
            for i in range (self.n_patches):
                gp = fp.create_group(f'patch_{i:02d}')
                gp.create_dataset(f'pair_inds', data=self.pair_inds[i])
                gp.create_dataset(f'pair_exp2phi', data=self.pair_exp2phi[i])
                gp.create_dataset(f'bins', data=self.bins[i])

    
    def load_pairs(self, filepath):
        
        self.pair_inds = []
        self.pair_exp2phi = []
        self.bins = []

        with h5py.File(filepath, 'r') as fp:
            self.nside = fp.attrs['nside']
            self.nbins = fp.attrs['nbins']
            self.theta_min = fp.attrs['theta_min']
            self.theta_max = fp.attrs['theta_max']
            self.patch_size = fp.attrs['patch_size']
            self.theta_Q = fp.attrs['theta_Q']
            self.n_patches = fp.attrs['n_patches']
            self.map_inds = fp['map_inds']
            self.phi_center = fp['phi_center'][:]
            self.theta_center = fp['theta_center'][:]
            
            for i in range(self.n_patches):
                gp = fp[f'patch_{i:02d}']
                self.pair_inds.append(gp['pair_inds'][:])
                self.pair_exp2phi.append(gp['pair_exp2phi'][:])
                self.bins.append(gp['bins'][:])
    
    def get_M_a(self, i, g1, g2, w):
        M_a = self.M_A_patch(self.Q_inds[i], self.Q_cos[i], self.Q_sin[i], self.Q_val[i], g1, g2, w, self.Q_patch_area[i])
        return M_a
    
    def calculate_M_a(self, g1, g2, w, threads=1):
        
        M_a = np.zeros(self.n_patches)
        
        with Pool(threads) as p:
            result = p.map(partial(self.get_M_a, g1=g1, g2=g2, w=w), range(self.n_patches))
        
        for i in range(self.n_patches):
            M_a[i] = result[i]
        
        self.M_a = M_a
        


class Correlation_CPU(Correlation):
    
    def __init__(self, nside, phi_center, theta_center, nbins=10, theta_min=10, theta_max=170, patch_size=90, theta_Q=90, mask=None, fastmath=True):
        super().__init__(nside, phi_center, theta_center, nbins, theta_min, theta_max, patch_size, theta_Q, mask, fastmath)
        self.xipm_patch = njit(fastmath=fastmath)(xipm_patch)
        
    def load_maps(self, g11, g21, g12, g22, w1, w2, flip_g1=True, flip_g2=False):
        self.g11 = g11
        self.g21 = g21
        self.g12 = g12
        self.g22 = g22
        self.w1 = w1
        self.w2 = w2
        
        if flip_g1: 
            self.g11 = -self.g11
            self.g12 = -self.g12
        if flip_g2: 
            self.g21 = -self.g21
            self.g22 = -self.g22
        
        if (g11 == g12).all() and (g21 == g22).all() and (w1 == w2).all():
            self.auto = True
            self.get_xipm = self.get_xipm_auto
        else:
            self.auto = False
            self.get_xipm = self.get_xipm_cross
        
    def get_xipm_auto(self, i):
        return self.xipm_patch(self.pair_inds[i], self.pair_exp2phi[i], self.bins[i], self.g11, self.g21, self.g12, self.g22, self.nbins)

    def get_xipm_cross(self, i):
        xip1, xim1 = self.xipm_patch(self.pair_inds[i], self.pair_exp2phi[i], self.bins[i], self.g11, self.g21, self.g12, self.g22, self.nbins)
        xip2, xim2 = self.xipm_patch(self.pair_inds[i], self.pair_exp2phi[i], self.bins[i], self.g12, self.g22, self.g11, self.g21, self.nbins)
        return (xip1 + xip2)/2, (xim1 + xim2)/2

    def calculate_2PCF(self, threads=1):
        
        xip, xim = np.zeros((self.n_patches, self.nbins)), np.zeros((self.n_patches, self.nbins))
        
        if threads > 1:
            with Pool(threads) as p:
                result = p.map(eval_func_tuple, zip(itertools.repeat(self.get_xipm), range(self.n_patches)))
        else:
            result = []
            for i in range(self.n_patches):
                result.append(self.get_xipm(i))
        
        for i in range(self.n_patches):
            xip[i] = result[i][0]
            xim[i] = result[i][1]
        
        self.xip = xip
        self.xim = xim
        
        return xip, xim

class Correlation_GPU(Correlation):
    
    def __init__(self, nside, phi_center, theta_center, nbins=10, theta_min=10, theta_max=170, patch_size=90, theta_Q=90, mask=None, fastmath=True, device=0):
        self.device = device
        super().__init__(nside, phi_center, theta_center, nbins, theta_min, theta_max, patch_size, theta_Q, mask, fastmath)
    
    def prepare_gpu(self):
        size = 0
        ninds = []
        for i in range(self.n_patches):
            patchsize = np.sum(self.bins[i])
            size += patchsize
            ninds.append(patchsize)

        first_patch_ind = np.append(0, np.cumsum(ninds))    
        temp_inds = np.zeros((2,size), dtype=np.uint32)
        temp_exp2phi = np.zeros((2,size), dtype=np.complex64)
        temp_bins = np.zeros((self.n_patches*self.nbins), dtype=np.float32)
        temp_bins_tot = np.zeros((self.n_patches * self.nbins), dtype=np.uint32)
             
        for i in range(self.n_patches):
            temp_inds[:,first_patch_ind[i]:first_patch_ind[i+1]] = self.pair_inds[i]
            temp_exp2phi[:,first_patch_ind[i]:first_patch_ind[i+1]] = self.pair_exp2phi[i]
            temp_bins[i*self.nbins:(i+1)*self.nbins] = self.bins[i]
            temp_bins_tot[i*self.nbins:(i+1)*self.nbins] = first_patch_ind[i] + self.bins[i].cumsum()
        temp_bins_tot = np.append([0], temp_bins_tot)
        
        self.inds_gpu = cp.asarray(temp_inds.astype(np.uint32))
        self.exp2phi_gpu = cp.asarray(temp_exp2phi.astype(np.complex64))
        self.bins_gpu = cp.asarray(temp_bins.astype(np.uint32))
        self.tot_bins_gpu = cp.asarray(temp_bins_tot.astype(np.uint32))
        self.ntotpairs = size
        
    def load_maps(self, g11, g21, g12, g22, w1, w2, flip_g1=True, flip_g2=False):
        self.g11 = cp.asarray(g11.astype(np.float32))
        self.g21 = cp.asarray(g21.astype(np.float32))
        self.g12 = cp.asarray(g12.astype(np.float32))
        self.g22 = cp.asarray(g22.astype(np.float32))
        self.w1 = cp.asarray(w1.astype(np.float32))
        self.w2 = cp.asarray(w2.astype(np.float32))
        
        if flip_g1: 
            self.g11 = -self.g11
            self.g12 = -self.g12
        if flip_g2: 
            self.g21 = -self.g21
            self.g22 = -self.g22
        
    def get_all_xipm(self):
        
        xip1, xim1 = cp.zeros((self.n_patches * self.nbins), dtype=cp.complex64), cp.zeros((self.n_patches * self.nbins), dtype=cp.complex64)
        xip2, xim2 = cp.zeros((self.n_patches * self.nbins), dtype=cp.complex64), cp.zeros((self.n_patches * self.nbins), dtype=cp.complex64)
        
        gt1 = ((self.g11[self.inds_gpu[0]]) + 1j* self.g21[self.inds_gpu[0]]) * self.exp2phi_gpu[0]
        gx1 = ((self.g12[self.inds_gpu[1]]) + 1j* self.g22[self.inds_gpu[1]]) * self.exp2phi_gpu[1]
        gt2 = ((self.g12[self.inds_gpu[0]]) + 1j* self.g22[self.inds_gpu[0]]) * self.exp2phi_gpu[0]
        gx2 = ((self.g11[self.inds_gpu[1]]) + 1j* self.g21[self.inds_gpu[1]]) * self.exp2phi_gpu[1]


        for bin in range(self.n_patches*self.nbins):
            xip1[bin] = cp.sum(gt1[self.tot_bins_gpu[bin]:self.tot_bins_gpu[bin+1]] * cp.conjugate(gx1[self.tot_bins_gpu[bin]:self.tot_bins_gpu[bin+1]]))/(self.bins_gpu[bin])
            xim1[bin] = cp.sum(gx1[self.tot_bins_gpu[bin]:self.tot_bins_gpu[bin+1]] * gt1[self.tot_bins_gpu[bin]:self.tot_bins_gpu[bin+1]])/(self.bins_gpu[bin])
            xip2[bin] = cp.sum(gt2[self.tot_bins_gpu[bin]:self.tot_bins_gpu[bin+1]] * cp.conjugate(gx2[self.tot_bins_gpu[bin]:self.tot_bins_gpu[bin+1]]))/(self.bins_gpu[bin])
            xim2[bin] = cp.sum(gx2[self.tot_bins_gpu[bin]:self.tot_bins_gpu[bin+1]] * gt2[self.tot_bins_gpu[bin]:self.tot_bins_gpu[bin+1]])/(self.bins_gpu[bin])
        
        xip = (xip1 + xip2).reshape((self.n_patches, self.nbins))/2
        xim = (xim1 + xim2).reshape((self.n_patches, self.nbins))/2
        
        return xip.real, xim.real