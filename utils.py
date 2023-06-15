import numpy as np
import math 

import lenstronomy.Util.util as util
import lenstronomy.Util.constants as const
from astropy.cosmology import default_cosmology
from lenstronomy.Cosmo.background import Background 
from lenstronomy.LensModel.Profiles.spp import SPP
from lenstronomy.LensModel.Profiles.sie import SIE
from lenstronomy.LensModel.Profiles.coreBurkert import CoreBurkert
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.lens_model import LensModel

from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
from lenstronomy.SimulationAPI.sim_api import SimAPI
import lenstronomy.Util.image_util as image_util

from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Data.psf import PSF
from lenstronomy.Util import simulation_util as sim_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.SimulationAPI.observation_api import SingleBand
from lenstronomy.SimulationAPI.data_api import DataAPI

from scipy.stats import norm, truncnorm, uniform

from paltas.Substructure import nfw_functions  
from paltas.Substructure.los_dg19 import LOSDG19 
from paltas.Utils import cosmology_utils



def rho_crit(cosmo, z):
    """
    critical density of cosmo at redshift z 
    :return: value in M_sol/Mpc^3
    """
    h = cosmo.H(z).value / 100.
    return 3 * h ** 2 / (8 * np.pi * const.G) * 10 ** 10 * const.Mpc / const.M_sun


def epl_m2thetae(m200, gamma, rho_c, s_crit): 
    '''
    Args: 
        m200 (float, np.array): m200's 
        gamma (float, np.array): density slopes 
        rho_c (float): critical density of universe 
        s_crit (float): sigma critical 
    
    Returns: 
        theta_E in arcsec corresponding to m200
    '''
    r = np.power(3*m200/(4*np.pi)/(200*rho_c), 1/3)
    rho0 = (3-gamma)/(4*np.pi)*m200/s_crit*r**(gamma-3)
    
    return SPP.rho2theta(rho0, gamma)


def epl_m200(theta_E, gamma, z_lens, z_source, cosmo): 
    lensCosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
    s_crit = lensCosmo.sigma_crit*lensCosmo.arcsec2phys_lens(1)**2
    rho_c = rho_crit(cosmo, z_lens)*lensCosmo.arcsec2phys_lens(1)**3
    
    rho0 = SPP.theta2rho(theta_E, gamma)
    
    r = np.power(3*rho0*s_crit/(3-gamma)/(200*rho_c), 1/gamma)
    
    #return SPP.mass_3d(r, rho_0, gamma)
    return 4/3 * np.pi * r**3 * (200*rho_c)


def get_sigmac(lensCosmo): 
    '''
    Args: 
        LensCosmo object in lenstronomy 
        
    Returns: 
        s_crit (float): sigma critical 
    '''
    # in Mpc 
    da_lens = lensCosmo.dd
    da_source = lensCosmo.ds
    da_ls = lensCosmo.dds

    Si_crit = const.c**2*da_source/(4*np.pi*const.G*da_lens*da_ls*const.Mpc)  # in kg/m^2 
    Si_crit = Si_crit/const.M_sun*const.Mpc**2  # in M_sun/Mpc^2 
    
    return Si_crit



def m_unif(x, m0=10**8, m1=10**11, beta=-1.9):
    '''
    taken from mathematica inverse of probability distribution of 0809.0898v1
    (x*m_max^(1+beta) + (1-x)*m_min^(1+beta))^(1/(1+beta))
    
    Args: 
        x (np.array): values pulled from U(0, 1) 
        m0 (float): min mass 
        m1 (float): max mass 
        beta (float): slope of SHMF 
        
    Returns: 
        array of masses pulled from SHMF 
    '''
    m0_b = np.power(m0, 1+beta)
    m1_b = np.power(m1, 1+beta)
    m = np.power(x * m1_b + (1-x)*m0_b, 1 / (1+beta))
    return m

def mass_to_concentration(m, dex, z=0.5):
    '''
    mass-concentration for nfw from https://academic.oup.com/mnras/article/441/4/3359/1209689 at z=0.5
    
    Args: 
        m (float, np.array): mass 
        dex (float): dex scatter from m to c 
        z (float): redshift 
        
    Returns: 
        concentration(s) corresponding to m 
    '''
    a = 0.520+(0.905-0.520)*math.exp(-0.617*z**1.21)
    b = -0.101+0.026*z
    return 10**(a + b*np.log10(m/(10**12/0.7)) + np.random.randn(len(m))*dex) 



# modified class of los halos based on a base class in paltas 
class LOSDG19_epl(LOSDG19):
    def __init__(self,gamma,gamma_width,los_parameters,main_deflector_parameters,source_parameters,cosmology_parameters):
        # Initialize the super class
        super().__init__(los_parameters,main_deflector_parameters,source_parameters,cosmology_parameters)
        self.gamma = gamma
        self.gamma_width = gamma_width
        
    def convert_to_lenstronomy_epl(self, z, z_masses, z_cart_pos): 
        """Converts the subhalo masses and position to EPL profiles
        for lenstronomy
        Args:
            z (float): The redshift for each of the halos
            z_masses (np.array): The masses of each of the halos that were drawn
            z_cart_pos (np.array): A n_los x 2D array of the position of the
                halos that were drawn
        Returns:
        ([string,...],[dict,...]): A tuple containing the list of models
        and the list of kwargs for each of the models.
        """
        
        nsub = len(z_masses) 
        
        z_source = self.source_parameters['z_source']
        lensCosmo = LensCosmo(z_lens=z, z_source=z_source, cosmo=self.cosmo.toAstropy())
        
        kpc_per_arcsecond = cosmology_utils.kpc_per_arcsecond(z,self.cosmo)
        cart_pos_ang = z_cart_pos / np.expand_dims(kpc_per_arcsecond,-1)

        # in M_sun and arcsec in lens plane 
        Si_crit = get_sigmac(lensCosmo)*(kpc_per_arcsecond/1e3)**2
        rho_c = rho_crit(self.cosmo.toAstropy(), z)*(kpc_per_arcsecond/1e3)**3
        
        g_loc, g_scale = self.gamma, self.gamma*self.gamma_width
        gammas = truncnorm.rvs((1.01-g_loc)/g_scale, (2.99-g_loc)/g_scale, loc=g_loc, scale=g_scale, size=nsub)
        e1s, e2s = np.random.uniform(-0.2, 0.2, (2, nsub)) 
        
        thetaes_los = epl_m2thetae(z_masses, gammas, rho_c, Si_crit)

        # Populate the parameters for each lens
        model_list = []
        kwargs_list = []

        for i in range(len(z_masses)):
            model_list.append('EPL')
            kwargs_list.append({'theta_E':thetaes_los[i], 'gamma':gammas[i],
                                'e1':e1s[i], 'e2':e2s[i],'center_x':cart_pos_ang[i,0],'center_y':cart_pos_ang[i,1]})
        
        return (model_list,kwargs_list)

    def convert_to_lenstronomy_nfw(self,z,z_masses,z_cart_pos,dex):
        """Converts the subhalo masses and position to truncated NFW profiles
        for lenstronomy
        Args:
            z (float): The redshift for each of the halos
            z_masses (np.array): The masses of each of the halos that were drawn
            z_cart_pos (np.array): A n_los x 2D array of the position of the
                halos that were drawn
        Returns:
        ([string,...],[dict,...]): A tuple containing the list of models
        and the list of kwargs for the truncated NFWs.
        """
        z_source = self.source_parameters['z_source']
        # First, draw a concentration for all our LOS structure from our mass
        # concentration relation
        concentration = mass_to_concentration(z_masses, dex, z=z)

        # Now convert our mass and concentration into the lenstronomy
        # parameters
        z_r_200 = nfw_functions.r_200_from_m(z_masses,z,self.cosmo)
        z_r_scale = z_r_200/concentration
        z_rho_nfw = nfw_functions.rho_nfw_from_m_c(z_masses,concentration,
                                                   self.cosmo,r_scale=z_r_scale)

        # Convert to lenstronomy units
        z_r_scale_ang, alpha_Rs = nfw_functions.convert_to_lenstronomy_NFW(
            z_r_scale,z,z_rho_nfw,z_source,self.cosmo)
        kpc_per_arcsecond = cosmology_utils.kpc_per_arcsecond(z,self.cosmo)
        cart_pos_ang = z_cart_pos / np.expand_dims(kpc_per_arcsecond,-1)

        # Populate the parameters for each lens
        model_list = []
        kwargs_list = []

        for i in range(len(z_masses)):
            model_list.append('NFW')
            kwargs_list.append({'alpha_Rs':alpha_Rs[i], 'Rs':z_r_scale_ang[i],
                                'center_x':cart_pos_ang[i,0],'center_y':cart_pos_ang[i,1]})

        return (model_list,kwargs_list)
    
        
    def negative_mass_sheet(self, numPix, deltapix, model_list, kwargs_list):
        '''
        Makes constant negative mass sheet given the model and image configs 
        
        Args: 
            numPix (int): number of pixels per size 
            deltapix (float): resolution per pixel 
            model_list: list of args for LensModel object 
            kwargs_list: kwargs correponding to each of the objects in model_list 
        
        Returns: 
            ['CONVERGENCE'], kwarg for negative mass sheet 
        '''
        
        x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltapix)
        lensModel_los = LensModel(model_list)
        kappa_los = lensModel_los.kappa(x_grid, y_grid, kwargs_list)
        
        mass_sheet = np.mean(kappa_los)
        
        return ['CONVERGENCE'], [{'kappa': -mass_sheet}] 
    
        
    def draw_los_epl(self, numPix, deltapix): 
        """Draws masses, concentrations,and positions for the los substructure 
        of a main lens halo.
        Returns:
            (tuple): A tuple of three lists: the first is the profile type for
            each los halo returned, the second is the lenstronomy kwargs
            for that halo, and the third is a list of redshift values for
            each profile.
            
        Notes:
            The returned lens model list includes terms to correct for
            the average deflection angle introduced from the los halos.
        """
        # Distribute line of sight substructure according to
        # https://arxiv.org/pdf/1909.02573.pdf. This also includes a
        # correction for the average deflection angle introduced by
        # the addition of the substructure.
        los_model_list = []
        los_kwargs_list = []
        los_z_list = []

        # Pull the paramters we need
        z_min = self.los_parameters['z_min']
        z_source = self.source_parameters['z_source']
        dz = self.los_parameters['dz']

        # Add halos from the starting reshift to the source redshift.
        # Note most of the calculations are done at z + dz/2, so you
        # want to stop at z_source-dz.
        z_range = np.arange(z_min,z_source-dz,dz)
        # Round the z_range to improve caching hits.
        z_range = list(np.round(z_range,2))

        # Iterate through each z and add the halos.
        for z in z_range:
            # Draw the masses and positions at this redshift from our
            # model
            z_masses = self.draw_nfw_masses(z)
            # Don't add anything to the model if no masses were drawn
            if z_masses.size == 0:
                continue
            z_cart_pos = self.sample_los_pos(z,len(z_masses))
            # Convert the mass and positions to lenstronomy models
            # and kwargs and append to our lists.
            model_list, kwargs_list = self.convert_to_lenstronomy_epl(z,z_masses,z_cart_pos)
            nms_list, kwargs_nms = self.negative_mass_sheet(numPix, deltapix, model_list, kwargs_list)
            
            los_model_list += model_list
            los_kwargs_list += kwargs_list
            
            los_model_list += nms_list
            los_kwargs_list += kwargs_nms
            
            los_z_list += [z+dz/2]*(len(model_list) + 1)

        return (los_model_list, los_kwargs_list, los_z_list)
    
    
    
    def draw_los_nfw(self, numPix, deltapix, c_dex):
        """Draws masses, concentrations,and positions for the los substructure
        of a main lens halo.
        Returns:
            (tuple): A tuple of three lists: the first is the profile type for
            each los halo returned, the second is the lenstronomy kwargs
            for that halo, and the third is a list of redshift values for
            each profile.
        Notes:
            The returned lens model list includes terms to correct for
            the average deflection angle introduced from the los halos.
        """
        # Distribute line of sight substructure according to
        # https://arxiv.org/pdf/1909.02573.pdf. This also includes a
        # correction for the average deflection angle introduced by
        # the addition of the substructure.
        los_model_list = []
        los_kwargs_list = []
        los_z_list = []

        # Pull the paramters we need
        z_min = self.los_parameters['z_min']
        z_source = self.source_parameters['z_source']
        dz = self.los_parameters['dz']

        # Add halos from the starting reshift to the source redshift.
        # Note most of the calculations are done at z + dz/2, so you
        # want to stop at z_source-dz.
        z_range = np.arange(z_min,z_source-dz,dz)
        # Round the z_range to improve caching hits.
        z_range = list(np.round(z_range,2))

        # Iterate through each z and add the halos.
        for z in z_range:
            # Draw the masses and positions at this redshift from our
            # model
            z_masses = self.draw_nfw_masses(z)
            # Don't add anything to the model if no masses were drawn
            if z_masses.size == 0: continue
            z_cart_pos = self.sample_los_pos(z,len(z_masses))
            # Convert the mass and positions to lenstronomy models
            # and kwargs and append to our lists.
            model_list, kwargs_list = self.convert_to_lenstronomy_nfw(z,z_masses,z_cart_pos,c_dex)
            nms_list, kwargs_nms = self.negative_mass_sheet(numPix, deltapix, model_list, kwargs_list)
            
            los_model_list += model_list
            los_kwargs_list += kwargs_list
            
            los_model_list += nms_list
            los_kwargs_list += kwargs_nms
            
            los_z_list += [z+dz/2]*(len(model_list) + 1)

        return (los_model_list, los_kwargs_list, los_z_list)

        
# ******************************************************************************
# Image Generation
# ******************************************************************************


def make_image(cosmo, 
               z_lens=0.2,
               z_source=0.6,
               numPix=100, 
               deltapix=0.08,
               main_lens_type='SIE',
               subhalo_type='EPL', 
               lens_light=True, 
               concentration_factor=1,  # concentration factor if subhalo_type=NFW, fix c=20 if set to 0
               dex=0, 
               max_sources=5,  # max number of sersics if sourceargs are not given 
               nsub=None,
               msubs=None,
               minmass=10**7,
               maxmass=10**10,
               beta=-1.9,
               gamma=None,    # optional pre-defined gammas for subhalos if nsub is given 
               pix_scale=0.5,   # threshold of brightness to determine positions to put subhalos 
               noise=2700,    # how much noise to add 
               nms=True,      # whether to add negative mass sheet 
               shear=0,       # external shear will be U(-shear, shear)
               multipole=False, # add multipole moments 
               sourceargs=None,   # includes source_model_list, kwargs_source matching lenstronomy 
               lensargs=None,     # kwargs matching lenstronomy; lenargs need to match main_lens_type 
               losargs=None):  
    
    side_length = numPix * deltapix   # side length in arcsec 
    # we set up a grid in coordinates and evaluate basic lensing quantities on it
    x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltapix)

    lensCosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
    mpc_per_arcsec_lens = lensCosmo.arcsec2phys_lens(1)

    # in M_sun and arcsec in lens plane 
    Si_crit = get_sigmac(lensCosmo)*mpc_per_arcsec_lens**2
    rho_c = rho_crit(cosmo, z_lens)*mpc_per_arcsec_lens**3

    if (subhalo_type == 'EPL'): 
        keys = ['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y']
    elif (subhalo_type == 'NFW'): 
        keys = ['Rs', 'alpha_Rs', 'center_x', 'center_y'] 
    elif (subhalo_type == 'coreBURKERT'): 
        keys = ['Rs', 'alpha_Rs', 'r_core', 'center_x', 'center_y']
    
    kwargs_source = []
    source_model_list = []
    lens_model_list = []
    kwargs_lens_list = []
    
    # make sersic source if sourceargs aren't given 
    if (sourceargs is None):
        N_Sources = np.random.randint(1 if max_sources==1 else 3, max_sources+1)

        for i in range(N_Sources):

            # the first source 
            if i == 0:
                phi_G, q = np.random.uniform(-np.pi, np.pi), np.random.uniform(1, 3)
                e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
                source_x, source_y = np.random.multivariate_normal(mean=[0, 0], cov=[[0.001, 0], [0, 0.001]])
                amp_power = 2.5 + np.random.random()**8  #np.random.uniform(2.5, 3.5)
                max_amp = np.power(10, amp_power)
                max_R = np.random.uniform(0.1,0.8)
                R = max_R
                amp = max_amp
            else:
                phi_G1, q1 = np.random.uniform(-np.pi, np.pi), np.random.uniform(1, 5)
                e1, e2 = param_util.phi_q2_ellipticity(phi_G1, q1)
                source_x, source_y = np.random.multivariate_normal(mean=[0, 0], cov=[[0.001, 0], [0, max_R**2]])
                source_x, source_y = np.cos(phi_G) * source_x + np.sin(phi_G) * source_y, np.cos(phi_G) * source_y - np.sin(phi_G) * source_x
                amp = np.random.uniform(0.1, max_amp)
                R = np.random.uniform(0.1, max_R*0.75)

                kwargs_sersic_source = {'amp': amp * R,
                                'R_sersic': R,
                                'n_sersic': np.random.uniform(0.9, 1.5),
                                'e1': e1,
                                'e2': e2,
                                'center_x': source_x,
                                'center_y': source_y
                               }

                source_model_list.append('SERSIC_ELLIPSE')
                kwargs_source.append(kwargs_sersic_source)
    
    else: 
        source_model_list, kwargs_source = sourceargs
    
    if (lensargs is None): 
        if main_lens_type == 'SIE':
            xlens, ylens = np.random.uniform(-0.25, 0.25, size=2)
            theta_E = np.random.uniform(2.8, 3.2)
            kwargs_lens_main = {'theta_E': theta_E, 
                                'e1': np.random.uniform(-0.2, 0.2),
                                'e2': np.random.uniform(-0.2, 0.2),
                                'center_x': xlens,
                                'center_y': ylens}
        elif main_lens_type == 'EPL':
            xlens, ylens = np.random.uniform(-0.25, 0.25, size=2)
            theta_E = np.random.uniform(2.8, 3.2)
            kwargs_lens_main = {'theta_E': theta_E, 
                                'gamma': np.random.normal(2, 0.2),
                                'e1': np.random.uniform(-0.2, 0.2),
                                'e2': np.random.uniform(-0.2, 0.2),
                                'center_x': xlens,
                                'center_y': ylens}
    else: 
        kwargs_lens_main = lensargs
        
    lens_model_list.append(main_lens_type)
    kwargs_lens_list.append(kwargs_lens_main)
    
    # add shear 
    if shear:
        kwargs_shear = {'gamma1': np.random.uniform(-shear, shear),
                        'gamma2': np.random.uniform(-shear, shear)}
        lens_model_list.append('SHEAR')
        kwargs_lens_list.append(kwargs_shear) 
        
    # add multipole moments 
    if multipole: 
        kwargs_mp = [{'m': 3, 
                     'a_m': np.random.uniform(0, 0.05),
                     'phi_m': np.random.uniform(-np.pi, np.pi),
                     'center_x': kwargs_lens_main['center_x'],
                     'center_y': kwargs_lens_main['center_y']}, 
                     {'m': 4, 
                     'a_m': np.random.uniform(0, 0.05),
                     'phi_m': np.random.uniform(-np.pi, np.pi),
                     'center_x': kwargs_lens_main['center_x'],
                     'center_y': kwargs_lens_main['center_y']}]
        lens_model_list += ['MULTIPOLE']*2
        kwargs_lens_list += kwargs_mp
        
    '''
    kwargs_model_mainlens = {'lens_model_list': lens_model_list,  # list of lens models to be used
                    'source_light_model_list': source_model_list,  # list of extended source models to be used
                    'z_lens': z_lens,
                    'z_source': z_source,
                    'cosmo': cosmo
                    }

    '''
    
    kwargs_psf = {'psf_type': 'GAUSSIAN',  # type of PSF model (supports 'GAUSSIAN' and 'PIXEL')
              'fwhm': 1.8*deltapix,  # full width at half maximum of the Gaussian PSF (in angular units)
              'pixel_size': deltapix  # angular scale of a pixel (required for a Gaussian PSF to translate the FWHM into a pixel scale)
             }
    psf_class = PSF(**kwargs_psf)
    
    # ACSF814W 
    # https://hst-docs.stsci.edu/wfc3ihb/chapter-6-uvis-imaging-with-wfc3/6-6-uvis-optical-performance
    kwargs_detector = {'pixel_scale':deltapix,'ccd_gain':1.58,'read_noise':3.0,
    'magnitude_zero_point':25.127, 'exposure_time':noise,'sky_brightness':21.83,
    'num_exposures':1,'background_noise':None
    }
    
    data_class = DataAPI(numpix=numPix,**kwargs_detector).data_class
    #kwargs_detector['pixel_scale'] = kwargs_detector.pop('pixel_width')
    single_band = SingleBand(**kwargs_detector)

    lens_model_class = LensModel(lens_model_list=lens_model_list, cosmo=cosmo)
    source_model_class = LightModel(light_model_list=source_model_list)
    
    kwargs_numerics = {'supersampling_factor': 1}
    imageModel_ml = ImageModel(data_class, psf_class, lens_model_class, source_model_class, kwargs_numerics=kwargs_numerics)
    
    im_ml = imageModel_ml.image(kwargs_lens_list, kwargs_source)
    
    '''
    # making this image to put the subhalos inside the ring 
    hst_ml = HST(band='WFC3_F160W', psf_type='GAUSSIAN')
    norbits_ml = hst_ml.kwargs_single_band() 
    norbits_ml['pixel_scale'] = deltapix
    norbits_ml['seeing'] = deltapix 
    Hub_ml = SimAPI(numpix=numPix,
                 kwargs_single_band=norbits_ml,
                 kwargs_model=kwargs_model_mainlens
                )

    hb_ml = Hub_ml.image_model_class(kwargs_numerics = {'point_source_supersampling_factor': 1})
    im_ml = hb_ml.image(kwargs_lens=kwargs_lens_list, kwargs_source=kwargs_source)
    '''
    # determine valid pixels for subhalo positioning 
    pix_max = np.max(im_ml)
    x_options, y_options = x_grid[im_ml.flatten() > pix_scale*pix_max], y_grid[im_ml.flatten() > pix_scale*pix_max]
    inds_choice = np.random.choice(range(len(x_options)), nsub)
    
    x = x_options[inds_choice] + np.random.uniform(-deltapix/2, deltapix/2, nsub)
    y = y_options[inds_choice] + np.random.uniform(-deltapix/2, deltapix/2, nsub)
    
    if (subhalo_type == 'EPL'): 
        # get other subhalo parameters 
        if (msubs is None): 
            msubs = m_unif(np.random.random(nsub), minmass, maxmass, beta=beta)
                      
        thetaes = epl_m2thetae(msubs, gamma, rho_c, Si_crit)
        e1s, e2s = np.random.uniform(-0.2, 0.2, (2, nsub)) 

        # subhalo parameter list
        vals = np.array([thetaes, gamma, e1s, e2s, x, y]).T
        
    elif (subhalo_type == 'NFW'): 
        if (msubs is None): 
            mass_nfw = m_unif(np.random.random(nsub), minmass, maxmass, beta=beta)
        else: 
            mass_nfw = msubs
        
        if (concentration_factor == 0):
            cs = 20*np.ones(len(mass_nfw)) 
        else: 
            cs = mass_to_concentration(mass_nfw, dex=dex, z=z_lens)*concentration_factor
            
        Rs_angle, alpha_Rs = lensCosmo.nfw_physical2angle(M=mass_nfw, c=cs)
        
        # subhalo parameter list
        vals = np.array([Rs_angle, alpha_Rs, x, y]).T
        
    elif (subhalo_type == 'coreBURKERT'): 
        if (msubs is None): 
            mass_burk = m_unif(np.random.random(nsub), minmass, maxmass, beta=beta)
        else: 
            mass_burk = msubs
            
        Rs_angle,_ = lensCosmo.nfw_physical2angle(M=mass_burk, c=mass_to_concentration(mass_burk))
        alpha_Rs = coreBurkert_mtoalpha(mass_burk, Rs_angle, Rs_angle, Si_crit, rho_c)
        
        vals = np.array([Rs_angle, alpha_Rs, Rs_angle, x, y]).T

    kwargs_subhalo_lens_list = []
    for val in vals: 
        kwargs_subhalo_lens_list.append(dict(zip(keys, val)))
            

    lensModel = LensModel(lens_model_list + [subhalo_type]*nsub)
    kappa = lensModel.kappa(x_grid, y_grid, kwargs_lens_list + kwargs_subhalo_lens_list)

    subhalo_lens_list = [subhalo_type]*nsub 
    
    if (nms):
        # negative mass sheet 
        lensModel_sh = LensModel([subhalo_type]*nsub)
        kappa_subhalos = lensModel_sh.kappa(x_grid, y_grid, kwargs_subhalo_lens_list)
        
        mass_sheet = np.mean(kappa_subhalos)
        subhalo_lens_list = subhalo_lens_list + ['CONVERGENCE']
        
        kwargs_subhalo_lens_list = kwargs_subhalo_lens_list + [{'kappa': -mass_sheet}]   # needs to be 'kappa' in updated lenstronomy 
        
    if lens_light: 
        lens_light_model_list = ['SERSIC_ELLIPSE']
    else: 
        lens_light_model_list = None 

    kwargs_lens_list += kwargs_subhalo_lens_list
    lens_model_list += subhalo_lens_list 
    
    kwargs_model = {'lens_model_list': lens_model_list,  # list of lens models to be used
                    'source_light_model_list': source_model_list,  # list of extended source models to be used
                    'lens_light_model_list': lens_light_model_list, 
                    'z_lens': z_lens,
                    'z_source': z_source,
                    'cosmo': cosmo
                    }

    if lens_light: 
        args_ll = {'magnitude': np.random.uniform(16, 19), 
                   'R_sersic': np.random.normal(0.8, 0.15), 
                   'n_sersic': np.random.uniform(1, 4), 
                   'e1': np.random.uniform(-0.1, 0.1), 
                   'e2': np.random.uniform(-0.1, 0.1), 
                   'center_x': kwargs_lens_main['center_x'], 
                   'center_y': kwargs_lens_main['center_y']}
        args_ll,_,_ = SimAPI(numpix=numPix, kwargs_single_band=kwargs_detector,kwargs_model=kwargs_model).magnitude2amplitude(kwargs_lens_light_mag=[args_ll])
        kwargs_light_lens = args_ll
        
        lens_light_model = LightModel(lens_light_model_list)
    else: 
        kwargs_light_lens = None 
        lens_light_model = None 
        
    
    lens_redshift_list = [z_lens]*len(kwargs_lens_list)
    
    multiplane = False 
    if losargs is not None: 
        los_model_list, los_kwargs_list, los_z_list = losargs
        
        lens_model_list += los_model_list 
        kwargs_lens_list += los_kwargs_list 
        lens_redshift_list += los_z_list 
        
        multiplane = True 
        
    lens_model_class = LensModel(lens_model_list,
                z_source=z_source,
                lens_redshift_list=lens_redshift_list,
                cosmo=cosmo, multi_plane=multiplane)
    
    kwargs_numerics = {'supersampling_factor': 1}
    imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model, kwargs_numerics=kwargs_numerics)
    
    im = imageModel.image(kwargs_lens_list, kwargs_source, kwargs_lens_light=kwargs_light_lens)
    
    if noise: 
        im += single_band.noise_for_model(im)
    
    
    '''
    # make image with noise 
    hst = HST(band='WFC3_F160W', psf_type='GAUSSIAN')
    norbits = hst.kwargs_single_band()
    norbits['pixel_scale'] = deltapix
    norbits['seeing'] = deltapix
    
    if (noise): 
        norbits['exposure_time'] = noise 
    
    Hub = SimAPI(numpix=numPix,
                 kwargs_single_band=norbits,
                 kwargs_model=kwargs_model
                )
    '''
    '''    
    hb_im = Hub.image_model_class(kwargs_numerics = {'point_source_supersampling_factor': 1})

    im = hb_im.image(kwargs_lens=kwargs_lens_list + kwargs_subhalo_lens_list, kwargs_source=kwargs_source, kwargs_lens_light=kwargs_light_lens)
    
    # add noise 
    if (noise): 
        hubnoise = Hub.noise_for_model(im)
        im = im + hubnoise
    '''
    out_dict = {'kwargs_model': kwargs_model,
                'kwargs_source': kwargs_source,
                'kwargs_lens': kwargs_lens_list, 
                'kwargs_lens_light': kwargs_light_lens, 
                'lens_redshifts': lens_redshift_list, 
                'Image': im, 
                'Image_ml': im_ml
               }
    
    return out_dict 



def make_image_from_args(kwargs_model, 
                          kwargs_source, 
                          kwargs_lens, 
                          kwargs_lens_light, 
                          numPix=100, 
                          deltapix=0.08, 
                          noise_exp=0):
    
    cosmo = kwargs_model['cosmo'] 
    z_lens = kwargs_model['z_lens'] 
    z_source = kwargs_model['z_source'] 
    
    side_length = numPix * deltapix   # side length in arcsec 
    # we set up a grid in coordinates and evaluate basic lensing quantities on it
    x_grid, y_grid = util.make_grid(numPix=numPix, deltapix=deltapix)
    
    lensCosmo = LensCosmo(z_lens=z_lens, z_source=z_source, cosmo=cosmo)
    
    # make image with noise 
    hst = HST(band='WFC3_F160W', psf_type='GAUSSIAN')
    norbits = hst.kwargs_single_band()
    norbits['pixel_scale'] = deltapix
    norbits['seeing'] = deltapix
    
    if (noise_exp): 
        norbits['exposure_time'] = noise_exp  # each orbit is 5400 secs 
    
    Hub = SimAPI(numpix=numPix,
                 kwargs_single_band=norbits,
                 kwargs_model=kwargs_model
                )
    
    hb_im = Hub.image_model_class(kwargs_numerics = {'point_source_supersampling_factor': 1})

    im = hb_im.image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source, kwargs_lens_light=kwargs_lens_light)
    
    # add noise 
    if (noise_exp): 
        hubnoise = Hub.noise_for_model(im)
        im = im + hubnoise
    
    out_dict = {'kwargs_model': kwargs_model,
                'kwargs_source': kwargs_source,
                'kwargs_lens': kwargs_lens, 
                'kwargs_lens_light': kwargs_lens_light, 
                'Image': im
               }
    
    return out_dict 