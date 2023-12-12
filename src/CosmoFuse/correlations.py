import numpy as np
import coord
from numba import njit
from multiprocessing import Pool
from .utils import pixel2RaDec, eval_func_tuple
from .correlation_helpers import rotate_shear, xipm_patch_auto, xipm_patch_cross, M_a_patch
import healpy as hp
import itertools
from functools import partial


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
        """
        This method finds all pairs of pixels at a given angular separation and their angles. 
        These are later used for calculating the 2PCFs.

        Args:
            patch_inds (ints): _description_
            ra (float): _description_
            dec (float): _description_
            return_pairs (bool, optional): Whether to return the result. Defaults to False.
        """
        
        all_inds, cos_2phi_all1, sin_2phi_all1, cos_2phi_all2, sin_2phi_all2, vartheta = [], [], [], [], [], []
        cos_vartheta = np.cos(np.subtract.outer(ra, ra))*np.multiply.outer(np.cos(dec),np.cos(dec)) + np.multiply.outer(np.sin(dec),np.sin(dec))
        dist = np.arccos(np.triu(cos_vartheta, k=1))        
        #dist = np.arccos(cos_vartheta)        
        

        for i in range(self.nbins):
            inds = np.where((dist > self.binedges[i]) & (dist < self.binedges[i+1]))
            pair_inds = np.array([patch_inds[inds[0]], patch_inds[inds[1]]])
            ra_pairs1 = ra[inds[0]]
            dec_pairs1 = dec[inds[0]]
            ra_pairs2 = ra[inds[1]]
            dec_pairs2 = dec[inds[1]]
            
            cos_2phi1, sin_2phi1 =rotate_shear(ra_pairs1, dec_pairs1, ra_pairs2, dec_pairs2)
            cos_2phi2, sin_2phi2 =rotate_shear(ra_pairs2, dec_pairs2, ra_pairs1, dec_pairs1)
            
            all_inds.append(pair_inds) # pair_inds=absolute pixel indices, inds=relative pixel indices
            cos_2phi_all1.append(cos_2phi1)
            sin_2phi_all1.append(sin_2phi1)
            cos_2phi_all2.append(cos_2phi2)
            sin_2phi_all2.append(sin_2phi2)
            vartheta.append(dist[inds[0],inds[1]])
            
        return all_inds, [cos_2phi_all1, sin_2phi_all1], [cos_2phi_all2, sin_2phi_all2]
    
    def __get_pairs_helper__(self, i):
        vec = hp.ang2vec(self.theta_center[i], self.phi_center[i])
        patch_inds = hp.query_disc(self.nside, vec=vec, radius=np.radians(self.patch_size/60))
        pix_inds = np.intersect1d(patch_inds, self.map_inds)
        ra, dec = pixel2RaDec(pix_inds, self.nside)
        inds, cos_sin_2phi_1, cos_sin_2phi_2 = self.get_pairs_patch(pix_inds, ra, dec)
        ninds = np.array([len(inds[i][0]) for i in range(self.nbins)])
        
        all_inds = np.zeros((2,ninds.sum()))
        all_cos_sin_2phi_1 = np.zeros((2,ninds.sum()))
        all_cos_sin_2phi_2 = np.zeros((2,ninds.sum()))
        for bin in range(self.nbins):
            all_inds[0,np.sum(ninds[:bin]):np.sum(ninds[:bin+1])] = inds[bin][0]
            all_inds[1,np.sum(ninds[:bin]):np.sum(ninds[:bin+1])] = inds[bin][1]
            all_cos_sin_2phi_1[0,np.sum(ninds[:bin]):np.sum(ninds[:bin+1])] = cos_sin_2phi_1[0][bin]
            all_cos_sin_2phi_1[1,np.sum(ninds[:bin]):np.sum(ninds[:bin+1])] = cos_sin_2phi_1[1][bin]
            all_cos_sin_2phi_2[0,np.sum(ninds[:bin]):np.sum(ninds[:bin+1])] = cos_sin_2phi_2[0][bin]
            all_cos_sin_2phi_2[1,np.sum(ninds[:bin]):np.sum(ninds[:bin+1])] = cos_sin_2phi_2[1][bin]
            
        return all_inds.astype(np.int32), all_cos_sin_2phi_1,  all_cos_sin_2phi_2, ninds
    
    
    
    def calculate_pairs_2PCF(self, threads=1):
        
        if threads == 1:
            pair_inds, pair_cos_sin_2phi_1, pair_cos_sin_2phi_2, bins = [], [], [], []
            for i in range(self.n_patches):
                result = self.__get_pairs_helper__(i)
                
                pair_inds.append(result[0])
                pair_cos_sin_2phi_1.append(result[1])
                pair_cos_sin_2phi_2.append(result[2])
                bins.append(result[3])

                print(f"Patch {i+1}/{self.n_patches} done.", end='\r', flush=True)
            print()
            
        else: 
            with Pool(threads) as p:
                pair_inds, pair_cos_sin_2phi_1, pair_cos_sin_2phi_2, bins =  list(map(list, zip(*p.map(self.__get_pairs_helper__, range(self.n_patches)))))
            
        self.pair_inds = pair_inds
        self.pair_cos_sin_2phi_1 = pair_cos_sin_2phi_1
        self.pair_cos_sin_2phi_2 = pair_cos_sin_2phi_2
        self.bins = bins


    def Q_T(self, theta):
        """The compensated filter used for aperture mass.

        Args:
            theta (float): Great Circle distance to center of filter.

        Returns:
            (float): Value of compensated filter.
        """
        
        theta_Q = np.radians(self.theta_Q/60)
        return theta**2/(4*np.pi*theta_Q**4)*np.exp(-theta**2/(2*theta_Q**2))

    def get_pairs_patch_M_a(self, pixels_RA_Q_patch, pixels_dec_Q_patch, Q_patch_center_RA, Q_patch_center_dec):

        cos_vartheta = np.cos(pixels_RA_Q_patch - Q_patch_center_RA)*np.cos(Q_patch_center_dec)*np.cos(pixels_dec_Q_patch) + np.sin(Q_patch_center_dec)*np.sin(pixels_dec_Q_patch)
        vartheta = np.arccos(cos_vartheta)
        sin_vartheta = np.sqrt(1-cos_vartheta**2)
        cos_phi = np.sin(pixels_RA_Q_patch - Q_patch_center_RA)*np.cos(pixels_dec_Q_patch) / sin_vartheta
        sin_phi = (np.cos(pixels_dec_Q_patch)*np.sin(Q_patch_center_dec) - np.sin(pixels_dec_Q_patch)*np.cos(Q_patch_center_dec)*np.cos(pixels_RA_Q_patch - Q_patch_center_RA)) / sin_vartheta
        cos_2phi = cos_phi*cos_phi - sin_phi*sin_phi
        sin_2phi = 2*sin_phi*cos_phi

        Q = self.Q_T(vartheta)

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
        pass
    
    def load_pairs(self, filepath):
        pass
    
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
            self.autocorrelation = True
            self.xipm_patch = njit(fastmath=self.fastmath)(xipm_patch_auto)
        else:
            self.autocorrelation = False
            self.xipm_patch = njit(fastmath=self.fastmath)(xipm_patch_cross)
    
    def get_xipm(self, i):
        return self.xipm_patch(self.pair_inds[i], self.pair_cos_sin_2phi_1[i], self.pair_cos_sin_2phi_2[i], self.bins[i], self.g11, self.g21, self.g12, self.g22, self.w1, self.w2, self.nbins)

        
    def calculate_2PCF(self, threads=1):
        
        xip, xim = np.zeros((self.n_patches, self.nbins)), np.zeros((self.n_patches, self.nbins))
        
        #with Pool(threads) as p:
            #result = p.map(eval_func_tuple, zip(itertools.repeat(self.get_xipm), range(self.n_patches)))
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
    
    def __init__(self, nside, phi_center, theta_center, nbins=10, theta_min=10, theta_max=170, patch_size=90, theta_Q=90, mask=None, fastmath=True):
        print("GPU not implemented yet.")
        print("Use Correlation_CPU instead.")
        super().__init__(nside, phi_center, theta_center, nbins, theta_min, theta_max, patch_size, theta_Q, mask, fastmath)
    
    def prepare_gpu(self):
        pass
        
