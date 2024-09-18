import numpy as np
from .correlations import Correlation
import healpy as hp
from scipy.special import binom
from sys import exit

try:
    import cupy as cp
except ImportError:
    print("Cupy not installed. Please install Cupy or use Correlation_GPU.")
    exit(1)

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
        temp_exp2phi = np.zeros((2,size), dtype=np.complex128)
        temp_bins = np.zeros((self.n_patches*self.nbins), dtype=np.uint32)
        temp_bins_tot = np.zeros((self.n_patches * self.nbins), dtype=np.uint32)

        for i in range(self.n_patches):
            temp_inds[:,first_patch_ind[i]:first_patch_ind[i+1]] = self.pair_inds[i]
            temp_exp2phi[:,first_patch_ind[i]:first_patch_ind[i+1]] = self.pair_exp2phi[i]
            temp_bins[i*self.nbins:(i+1)*self.nbins] = self.bins[i]
            temp_bins_tot[i*self.nbins:(i+1)*self.nbins] = first_patch_ind[i] + self.bins[i].cumsum()
        temp_bins_tot = np.append([0], temp_bins_tot)
        
        self.inds_gpu = cp.asarray(temp_inds)
        self.exp2phi_gpu = cp.asarray(temp_exp2phi)
        self.bins_gpu = cp.asarray(temp_bins)
        self.tot_bins_gpu = cp.asarray(temp_bins_tot)
        self.ntotpairs = size
        
    def load_maps(self, g11, g21, g12, g22, w1, w2, sumofweights=None, flip_g1=True, flip_g2=False):
        self.g11 = cp.asarray(g11)
        self.g21 = cp.asarray(g21)
        self.g12 = cp.asarray(g12)
        self.g22 = cp.asarray(g22)
        self.w1 = cp.asarray(w1)
        self.w2 = cp.asarray(w2)
        if sumofweights is None:
            self.sumofweights = cp.add.reduceat(self.w1[self.inds_gpu[0]] * self.w2[self.inds_gpu[1]], self.tot_bins_gpu[:-1])
        else:
            self.sumofweights = sumofweights
            
        if flip_g1: 
            self.g11 = -self.g11
            self.g12 = -self.g12
        if flip_g2: 
            self.g21 = -self.g21
            self.g22 = -self.g22
            
        
    def xipm(self, g11, g21, g12, g22, w1, w2, sumofweights):
        
        g2 = w1[self.inds_gpu[0]]*((g11[self.inds_gpu[0]]) + 1j* g21[self.inds_gpu[0]]) * self.exp2phi_gpu[0]
        g1 = w2[self.inds_gpu[1]]*((g12[self.inds_gpu[1]]) + 1j* g22[self.inds_gpu[1]]) * self.exp2phi_gpu[1]

        xip = ((cp.add.reduceat(g1 * cp.conjugate(g2), self.tot_bins_gpu[:-1]))/sumofweights).reshape((self.n_patches, self.nbins))
        xim = ((cp.add.reduceat(g1 * g2, self.tot_bins_gpu[:-1]))/sumofweights).reshape((self.n_patches, self.nbins))
        
        return xip.real, xim.real   
        
    def __xipm(self):
        
        g2 = self.w1[self.inds_gpu[0]]*((self.g11[self.inds_gpu[0]]) + 1j* self.g21[self.inds_gpu[0]]) * self.exp2phi_gpu[0]
        g1 = self.w2[self.inds_gpu[1]]*((self.g12[self.inds_gpu[1]]) + 1j* self.g22[self.inds_gpu[1]]) * self.exp2phi_gpu[1]

        xip = ((cp.add.reduceat(g1 * cp.conjugate(g2), self.tot_bins_gpu[:-1]))/self.sumofweights).reshape((self.n_patches, self.nbins))
        xim = ((cp.add.reduceat(g1 * g2, self.tot_bins_gpu[:-1]))/self.sumofweights).reshape((self.n_patches, self.nbins))
        
        return xip.real, xim.real
    
    def __xipm_c(self):
        
        g1 = self.w1[self.inds_gpu[0]]*((self.g11[self.inds_gpu[0]]) + 1j* self.g21[self.inds_gpu[0]]) * self.exp2phi_gpu[0]
        g2 = self.w2[self.inds_gpu[1]]*((self.g12[self.inds_gpu[1]]) + 1j* self.g22[self.inds_gpu[1]]) * self.exp2phi_gpu[1]

        xip = ((cp.add.reduceat(g1 * cp.conjugate(g2), self.tot_bins_gpu[:-1]))/self.sumofweights).reshape((self.n_patches, self.nbins))
        xim = ((cp.add.reduceat(g1 * g2, self.tot_bins_gpu[:-1]))/self.sumofweights).reshape((self.n_patches, self.nbins))
        
        return xip.real, xim.real
    
    
    def get_all_xipm(self):
        xip1, xim1 = self.__xipm()
        xip2, xim2 = self.__xipm_c()
        
        return (xip1 + xip2)/2, (xim1 + xim2)/2
    
    def get_full_tomo(self, shear_maps, w, sumofweights, flip_g1=False, flip_g2=False):
        
        nzbins = shear_maps.shape[0]
        nzbin_combs = int(binom(nzbins+1, 2))
        shear_maps_gpu = cp.asarray(shear_maps)
        w_gpu = cp.asarray(w)
        sumofweights_gpu = cp.asarray(sumofweights)
        
        g1, g2 = 1, 1
        if flip_g1:
            g1 = -1
        if flip_g2:
            g2 = -1
        
        M_ap = np.zeros([nzbins, self.n_patches])
        xim1 = cp.zeros([nzbin_combs, self.n_patches, self.nbins])
        xim2 = cp.zeros([nzbin_combs, self.n_patches, self.nbins])
        xip1 = cp.zeros([nzbin_combs, self.n_patches, self.nbins])
        xip2 = cp.zeros([nzbin_combs, self.n_patches, self.nbins])
        
        k=0
        for i in range(nzbins):
            M_ap[i] = self.get_M_a(g1*shear_maps[i,0], g2*shear_maps[i,1], w[i])
            for j in range(i, nzbins):
                if i == j:
                    xip1[k], xim1[k] = self.xipm(g1*shear_maps_gpu[i,0], g2*shear_maps_gpu[i,1], g1*shear_maps_gpu[j,0], g2*shear_maps_gpu[j,1], w_gpu[i], w_gpu[j], sumofweights_gpu[0,k])
                    xip2[k], xim2[k] = xip1[k], xim1[k]
                else:
                    xip1[k], xim1[k] = self.xipm(g1*shear_maps_gpu[i,0], g2*shear_maps_gpu[i,1], g1*shear_maps_gpu[j,0], g2*shear_maps_gpu[j,1], w_gpu[i], w_gpu[j], sumofweights_gpu[0,k])
                    xip2[k], xim2[k] = self.xipm(g1*shear_maps_gpu[j,0], g2*shear_maps_gpu[j,1], g1*shear_maps_gpu[i,0], g2*shear_maps_gpu[i,1], w_gpu[j], w_gpu[i], sumofweights_gpu[1,k])
                k += 1
                
        xip = (xip1 + xip2)/2
        xim = (xim1 + xim2)/2
        
        return M_ap, xip.get(), xim.get()
        
        
        
class Correlation_GPU_lowmem(Correlation_GPU):
    
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
        temp_bins = np.zeros((self.n_patches*self.nbins), dtype=np.uint32)
        temp_bins_tot = np.zeros((self.n_patches * self.nbins), dtype=np.uint32)

        for i in range(self.n_patches):
            temp_inds[:,first_patch_ind[i]:first_patch_ind[i+1]] = self.pair_inds[i]
            temp_exp2phi[:,first_patch_ind[i]:first_patch_ind[i+1]] = self.pair_exp2phi[i]
            temp_bins[i*self.nbins:(i+1)*self.nbins] = self.bins[i]
            temp_bins_tot[i*self.nbins:(i+1)*self.nbins] = first_patch_ind[i] + self.bins[i].cumsum()
        temp_bins_tot = np.append([0], temp_bins_tot)
        
        self.inds_gpu = cp.asarray(temp_inds)
        self.exp2phi_gpu = cp.asarray(temp_exp2phi)
        self.bins_gpu = cp.asarray(temp_bins)
        self.tot_bins_gpu = cp.asarray(temp_bins_tot)
        self.ntotpairs = size
    
    
    def xipm(self, g11, g21, g12, g22, w1, w2, sumofweights):
        
        g2 = w1[self.inds_gpu[0]]*((g11[self.inds_gpu[0]]) + 1j* g21[self.inds_gpu[0]]) * self.exp2phi_gpu[0]
        g1 = w2[self.inds_gpu[1]]*((g12[self.inds_gpu[1]]) + 1j* g22[self.inds_gpu[1]]) * self.exp2phi_gpu[1]

        xip, xim = cp.zeros((self.n_patches*self.nbins)), cp.zeros((self.n_patches*self.nbins))
        
        for tot_ind in range(self.n_patches*self.nbins):
            xip[tot_ind] = cp.sum(g1[self.tot_bins_gpu[tot_ind]:self.tot_bins_gpu[tot_ind+1]] * cp.conjugate(g2[self.tot_bins_gpu[tot_ind]:self.tot_bins_gpu[tot_ind+1]]))/(sumofweights[tot_ind]).real
            xim[tot_ind] = cp.sum(g1[self.tot_bins_gpu[tot_ind]:self.tot_bins_gpu[tot_ind+1]] * g2[self.tot_bins_gpu[tot_ind]:self.tot_bins_gpu[tot_ind+1]])/(sumofweights[tot_ind]).real
        
        return xip.real, xim.real
    
    
    def get_full_tomo(self, shear_maps, w, sumofweights, flip_g1=False, flip_g2=False):
        
        nzbins = shear_maps.shape[0]
        nzbin_combs = int(binom(nzbins+1, 2))
        shear_maps_gpu = cp.asarray(shear_maps.astype(np.float16))
        w_gpu = cp.asarray(w.astype(np.float16))
        sumofweights_gpu = cp.asarray(sumofweights.astype(np.float16))
        
        g1, g2 = 1, 1
        if flip_g1:
            g1 = -1
        if flip_g2:
            g2 = -1
        
        M_ap = np.zeros([nzbins, self.n_patches])
        xim1 = cp.zeros([nzbin_combs, self.n_patches * self.nbins])
        xim2 = cp.zeros([nzbin_combs, self.n_patches * self.nbins])
        xip1 = cp.zeros([nzbin_combs, self.n_patches * self.nbins])
        xip2 = cp.zeros([nzbin_combs, self.n_patches * self.nbins])
        
        k=0
        for i in range(nzbins):
            M_ap[i] = self.get_M_a(g1*shear_maps[i,0], g2*shear_maps[i,1], w[i])
            for j in range(i, nzbins):
                if i == j:
                    xip1[k], xim1[k] = self.xipm(g1*shear_maps_gpu[i,0], g2*shear_maps_gpu[i,1], g1*shear_maps_gpu[j,0], g2*shear_maps_gpu[j,1], w_gpu[i], w_gpu[j], sumofweights_gpu[0,k])
                    xip2[k], xim2[k] = xip1[k], xim1[k]
                else:
                    xip1[k], xim1[k] = self.xipm(g1*shear_maps_gpu[i,0], g2*shear_maps_gpu[i,1], g1*shear_maps_gpu[j,0], g2*shear_maps_gpu[j,1], w_gpu[i], w_gpu[j], sumofweights_gpu[0,k])
                    xip2[k], xim2[k] = self.xipm(g1*shear_maps_gpu[j,0], g2*shear_maps_gpu[j,1], g1*shear_maps_gpu[i,0], g2*shear_maps_gpu[i,1], w_gpu[j], w_gpu[i], sumofweights_gpu[1,k])
                k += 1
                
        xip = ((xip1 + xip2)/2).reshape((nzbin_combs, self.n_patches, self.nbins))
        xim = ((xim1 + xim2)/2).reshape((nzbin_combs, self.n_patches, self.nbins))
        
        return M_ap, xip.get(), xim.get()