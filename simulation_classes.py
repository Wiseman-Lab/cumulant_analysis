import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import tifffile as tif

"""
convolving with the gaussian + ball kernel doesn't quite work.
"""

Generator = np.random.default_rng()

    
def gaussian(x, y, w0):
    return np.exp(-2*(x**2 + y**2) / w0**2)

class Particles():
    def __init__(self, N=1, D = 0.1, spectrum=[1]) -> None:
        self.N = N # particle density/um^2
        self.D = D # diffusion constant (um^2/s)
        self.spectrum = np.atleast_1d(spectrum)
        self.xy = None
        self.parentimg = None
    
    def set_random_pos(self):
        # set the appropriate number ofrandom positions for this species 
        self.xy = np.zeros(shape=( 2, int(self.parentimg.pix_len**2 * self.parentimg.w * self.parentimg.h * self.N) ))     
        self.xy[0] = Generator.uniform(0, self.parentimg.h, self.xy.shape[1])
        self.xy[1] = Generator.uniform(0, self.parentimg.w, self.xy.shape[1])

    def set_fixed_pos(self, x, y):
        self.xy = np.zeros((2, len(x)))
        self.xy[0] = x
        self.xy[1] = y

    def update_pos(self, dt):
        # update particle positions based on diffusion model for given dt.
        dxy = Generator.normal(0, 2*self.D*dt, size=self.xy.shape)
        self.xy = self.xy + (dxy/self.parentimg.pix_len) #scaled by pixel length.
        #periodic boundary conditions
        self.xy[0, self.xy[0] < 0 ] = self.parentimg.h + self.xy[0, self.xy[0] < 0 ] 
        self.xy[1, self.xy[1] < 0 ] = self.parentimg.w + self.xy[1, self.xy[1] < 0 ]
        self.xy[0, self.xy[0] > self.parentimg.h ] = self.xy[0, self.xy[0] > self.parentimg.h ] - self.parentimg.h 
        self.xy[1, self.xy[1] > self.parentimg.w ] = self.xy[1, self.xy[1] > self.parentimg.w ] - self.parentimg.w 
        
    def get_image(self):
        if self.parentimg.random_pos:
            pixels = np.histogram2d(self.xy[0], self.xy[1], bins=[(self.parentimg.h), (self.parentimg.w)], range=[[0,self.parentimg.h], [0,self.parentimg.w]])[0]
        else:
            pixels = np.histogram2d(self.xy[0], self.xy[1], bins=[(self.parentimg.h), (self.parentimg.w)], range=[[0,self.parentimg.h], [0,self.parentimg.w]])[0]
        img = np.zeros((len(self.spectrum), pixels.shape[0], pixels.shape[1]))
        for i in range(len(self.spectrum)):
            if i == 0:
                img[i, :, :] = signal.fftconvolve(pixels, self.parentimg.kernel, mode='same')
            if i==1:
                img[i, :, :] = signal.fftconvolve(pixels, self.parentimg.kernel2, mode='same')                
            img[i, :, :] = img[i, :, :] * self.spectrum[i]
        img = img[:, 20:-20, 20:-20]
        img[img<0] = 0 #sometimes sparse images get small negative values.
        #poisson filter
        if self.parentimg.poisson:
            img = img + Generator.random(img.shape)*1e-16 # poisson(0) is not allowed so add very smal noise
            img = Generator.poisson(img)
            
        return img


    def get_confocal_image(self):
        xpix = np.arange(0, self.parentimg.h) + 0.5
        ypix = np.arange(0, self.parentimg.w) + 0.5
        xco, yco = np.meshgrid(xpix, ypix, indexing='ij')

        # ### nested for loops way ###
        # img = np.zeros((len(self.spectrum), len(xpix), len(xpix)))
        # for i in range(self.parentimg.h):
        #     for j in range(self.parentimg.w):
        #         xdiffs = xco[i, j] - self.xy[0, :]
        #         ydiffs = yco[i, j] - self.xy[1, :]
        #         for k in range(len(self.spectrum)):
        #             img[k, :, :] = np.sum(gaussian(xdiffs, ydiffs, self.parentimg.r[k]), axis=0)      
        #             img[k, :, :] = img[k, :, :] * self.spectrum[k]

        ### super memory intensive but technically vectorized way... ###
        xdiffs = np.ones((self.xy.shape[-1], len(xpix), len(ypix))) * xco - self.xy[0, :, np.newaxis, np.newaxis]
        ydiffs = np.ones((self.xy.shape[-1], len(xpix), len(ypix))) * yco - self.xy[1, :, np.newaxis, np.newaxis]

        img = np.zeros((len(self.spectrum), len(xpix), len(xpix)))
        for i in range(len(self.spectrum)):
            img[i, :, :] = np.sum(gaussian(xdiffs, ydiffs, self.parentimg.r[i]/self.parentimg.pix_len), axis=0)      
            img[i, :, :] = img[i, :, :] * self.spectrum[i]

        img = img[:, 20:-20, 20:-20]
        img[img<0] = 0 #sometimes sparse images get small negative values.
        #poisson filter
        if self.parentimg.poisson:
            img = img + Generator.random(img.shape)*1e-16 # poisson(0) is not allowed so add very smal noise
            img = Generator.poisson(img)
        return img


    # def get_pixel(self, x, y):
    #     """
    #     THIS DOESNT WORK nicely yet/
    #     There is definitely a clever way to do this but I can't figure out how to scale things with a gaussian psf on the single pixel level.
    #     just do the whole image capture and only return that pixel.
    #     """
    #     pixels = np.histogram2d(self.xy[:, 0], self.xy[:, 1], bins=[(self.parentimg.h), (self.parentimg.w)], range=[[0,self.parentimg.h], [0,self.parentimg.w]])[0]
    #     img = signal.convolve2d(pixels, self.parentimg.kernel, mode='same') * self.brightness ########### maybe I have to divide by the maximum of the gaussian kernel here??
    #     img[img<0] = 0 #sometimes sparse images get small negative values.
    #     img = img[20:-20, 20:-20]
    #     #poisson filter
    #     if self.parentimg.poisson:
    #         img = img + Generator.random(img.shape)*1e-16 # poisson(0) is not allowed so add very smal noise
    #         img = Generator.poisson(img)
    #     if len(self.spectrum) > 1:
    #         colorimg = np.zeros((len(self.spectrum), img.shape[0], img.shape[1]))
    #         for (x, y), val in np.ndenumerate(img):
    #             colorimg[:, x, y] = Generator.multinomial(val, pvals=self.spectrum)
    #         img = colorimg

    #     return img[:, x, y]


class Image():
    def __init__(self, particles=[], w=256, h=256, gaussian_radius=[0.5], raster=False, pix_len=0.1, numchannels=1, background=[0], poisson=False, fmap=None, pixel_dwell_time=None, line_time=None, random_pos=True, imtype=np.float64) -> None:
        self.r = gaussian_radius #gaussian e^-2 radius of psf (um)
        self.w = w+40 # width in pixels
        self.h = h+40 # height in pixels ##################################### these are pre-padded. When you extract the image, need to go from [20:-20, 20:-20]
        self.raster = raster # raster scan or full image scan (not for use yet)
        self.pix_len = pix_len # side length of 1 (square pixel)
        self.particles = particles # list of all particle species
        self.numchannels = numchannels
        self.pixel_dwell_time = pixel_dwell_time      
        self.line_time = line_time # (for raster mode: NOT for use yet)
        self.background = background # scaling factor for background noise
        self.background_sum = np.sum(np.atleast_1d(background))
        self.fmap = fmap
        self.kernel = None 
        self.kernel2 = None
        self.current_img = None
        self.poisson = poisson
        self.g_scale = None
        self.random_pos = random_pos
        self.imtype = imtype
        if self.random_pos:
            self.set_particles_rand()
        self.make_kernel()
        for species in self.particles:
            species.parentimg = self
        

    def set_particles_rand(self):
        for species in self.particles:
            species.parentimg = self
            species.set_random_pos()

    def set_particles_fixed(self, xpos, ypos):
        for i, species in enumerate(self.particles):
            species.set_fixed_pos(xpos[i], ypos[i])

    def make_kernel(self, correct=False): 
        """
        this works as expected.
        """
        # make kernel for 2d convolution that combines gaussian with proper radius to capture out of pixel fluorescence.
        self.kernel = self.gaussian_kernel(self.r[0], correct=correct)
        self.g_scale = self.kernel.max()
        if len(self.r) == 2:
            self.kernel2 = self.gaussian_kernel(self.r[1], correct=correct)
    
    def gaussian_kernel(self, r, correct=False):
        """
        This works like I expect it to.

        r should be an element of the list self.r

        N.B. 
            This is the wrong kind of normalized gaussian. fluorescence microscopy definitions use 
            a form like exp[-2(r^2)/s^2] with s the 1/e^2 radius of the gaussian beam 
            Note exp[-2(0^2)/s^2] = 1
        """
        l = int(r/self.pix_len) * 4 # l is half side length -1, *4 just makes the kernel big enough.
        x = np.arange(-l, l+1)
        if correct:
            xgaus = np.exp( -2 * ( x**2 ) / ( (r/self.pix_len)**2 ) )
            gaus = np.outer(xgaus, xgaus)
            return gaus / np.max(gaus)
        else:
            xgaus = np.exp( -( x**2 ) / ( 2 * (r/self.pix_len/2)**2 ) )
            gaus = np.outer(xgaus, xgaus)
            return gaus / np.sum(gaus)
    

    def update_particles(self, dt):
        for species in self.particles:
            species.update_pos(dt)


    def get_img(self, confocal=False):
        # get a normal image
        img = np.zeros((self.numchannels, self.h-40, self.w-40))
        if confocal:
            for species in self.particles:
                img = img + species.get_confocal_image() # always assume channels match between species and image specifications
        else:
            for species in self.particles:
                img = img + species.get_image() # always assume channels match between species and image specifications
        new_image = np.squeeze(img)

        if self.fmap is not None:
            if len(new_image.shape) > 2:
                for im in new_image:
                    # maps need to have the same shape as the image frame
                    im[self.fmap == 0] = 0
            else:
                new_image[self.fmap == 0] = 0
        self.current_img = new_image

        if self.background_sum > 0:
            self.add_noise()

    def add_noise(self):
        # if setting multiple channel noise, then you should already have multichannel images...
        if hasattr(self.background, '__len__'):
            for i in range(self.current_img.shape[0]):
                self.current_img[i] = self.current_img[i] + Generator.poisson(self.background[i], self.current_img[i].shape)
        else:
            self.current_img = self.current_img + Generator.poisson(self.background, self.current_img.shape)

    def give_img(self):
        return self.current_img.astype(self.imtype)
    
    def get_raster_img(self):
        """
        NOT for use yet
        """
        img = np.zeros(self.numchannels, self.h-40, self.w-40)
        for row in range(self.h-40):
            for col in range(self.w-40):
                pixel = np.zeros(self.numchannels)
                for species in self.particles:
                    pixel = pixel + species.get_pixel(row, col) # multichannel pixel value for each species.
                self.update_particles(self.pixel_dwell_time) ### This simulation pretends that lines can be instantaneously switched between (instantaneous jump to start of next row).
                img[:, row, col] = pixel
        self.current_img = np.squeeze(img)
    
    def save_img(self, path):
        tif.imwrite(path, self.current_img)
    
    def show_image(self, title=None, cmaps=['bone', 'gist_heat']):
        if len(self.current_img.shape) > 2:
            for ich in range(self.current_img.shape[0]):
                plt.figure()
                if ich == 0:
                    plt.imshow(self.current_img[ich, :, :], cmap = cmaps[0])
                elif ich == 1:
                    plt.imshow(self.current_img[ich, :, :], cmap = cmaps[1])
                plt.xticks([])
                plt.yticks([])
                # plt.xticks(np.arange(0, 256, 50), np.arange(0, 256, 50)*0.1)
                # plt.yticks(np.arange(0, 256, 50), np.arange(0, 256, 50)*0.1)
                # plt.xlabel(r'x ($\mu$m)')
                # plt.ylabel(r'y ($\mu$m)')
                plt.colorbar()
                print(np.mean(self.current_img[ich, :, :]))
                if title is None:
                    plt.title(f'Channel {ich+1}')
                else:
                    plt.title(title + ' ' + f'Channel {ich+1}')

        else:
            plt.figure()
            plt.imshow(self.current_img, cmap=cmaps[0])
            plt.colorbar()
            print(np.mean(self.current_img))
            if title is not None:
                plt.title(title)

        plt.show()


def quick_sim(free1density, free2density, complex_density, brightness=100, noise=1):
    r = free1density
    g = free2density
    rg = complex_density

    R = Particles(r, D=0, spectrum=[brightness, 0])
    G = Particles(g, D=0, spectrum=[0, brightness])    
    RG = Particles(rg, D=0, spectrum=[brightness, brightness])
    im = Image(particles=[R, G, RG], w=256, h=256, gaussian_radius=0.5, poisson=True, background=noise)
    im.get_img()
    im.show_image()
    return im.give_img()