import math
import numpy
import pygame

"""
Based on the BalanceTrials.m file and function found in Psychotoolbox 3.0 for Matlab.
http://psychtoolbox.org/HomePage
"""

def balanceTrials(n_trials, randomize, factors, use_type='int'):
    n_factors = len(factors)
    n_levels = [0] * n_factors
    min_trials = 1.0 #needs to be float or the later ceiling operation will fail
    for f in range(0, n_factors):
        n_levels[f] = len(factors[f])
        min_trials *= n_levels[f] #simulates use of prod(n_levels) in the original code
    
    N = math.ceil(n_trials / min_trials)
    
    output = []
    len1 = min_trials
    len2 = 1
    index = numpy.random.uniform(0, 1, N * min_trials).argsort()
    
    for level, factor in zip(n_levels, factors):
        len1 /= level
        factor = numpy.array(factor, dtype=use_type)
        
        out = numpy.kron(numpy.ones((N, 1)), numpy.kron(numpy.ones((len1, len2)), factor).reshape(min_trials,)).astype(use_type).reshape(N*min_trials,)
        
        if randomize:
            out = [out[i] for i in index]
        
        len2 *= level
        output.append(out)
    
    return output

"""
reimplementation of the mat2gray functionality from Matlab
"""
def mat2gray(A):
    A = A.astype('float64')
    minimum, maximum = numpy.min(A), numpy.max(A)
    
    I = None
    
    if minimum == maximum:
        I = A
    else:
        delta = 1.0 / (maximum - minimum)
        I = (delta * A - minimum * delta)
    
    return numpy.clip(I, 0.0, 1.0)
    
"""
Based on code from Aaron Johnsons GratingThresholdIm implementation
"""

def color_rms(image, rms):
    output = numpy.copy(image).astype('float64')
    for d in range(0, 3):
        n = output[:, :, d]
        n = (n / 127.0) - 1.0
        n = n / numpy.max(numpy.abs(n))/2.0
        rms_altered = numpy.std(n + 0.5) / numpy.mean(n + 0.5)
        rms_alt_scale = (rms * 3.0) / rms_altered
        n = n * rms_alt_scale
        n = n * 128.0 + 128.0
        output[:,:,d] = n
    output = numpy.clip(output, 0, 255).astype('uint8')
    return output

from collections import namedtuple
VIS_DATA = namedtuple("VisData", ['resolution', 'midpoint', 'size', 'vdist'], verbose=False, rename=False)
SPATIAL_DATA = namedtuple("SpacialData", ['ecc', 'sf', 'rot'])
FREQ_DATA = namedtuple("FreqData", ['pixels_per_degree', 'gabor_diameter', 'xf', 'yf', 'guassian', 'ramp', 'grating', 'g'])

GABOR_DEF = namedtuple("GaborDef", ['original', 'matrix', 'rms_matrix', 'avg_matrix'])

GABOR_DATA = namedtuple("GaborData", ['position', 'size', 'radius', 'old_patch', 'new_patch'])

def load_matrices(source, resolution=(1024, 768), is_rms=False):
    s_image_original = pygame.image.load(source)
    s_image_original = pygame.transform.smoothscale(s_image_original, resolution)
    s_image_mat = pygame.surfarray.array3d(s_image_original)
    s_image_rms_mat = s_image_mat.copy() if is_rms else color_rms(s_image_mat, 0.2)
    s_avg_mat = numpy.mean(s_image_rms_mat, axis=2)
    return GABOR_DEF._make([s_image_original, s_image_mat, s_image_rms_mat, s_avg_mat])

def load_spacial_data(
    (resolution, midpoint, size, vdist),
    (ecc, sf, orientation),
    sigma=0.5):
    
    pixel_size = [x/y for x,y in zip(size, resolution)]
    
    degrees_per_pixel = [2.0 * math.atan(size/(2.0*vdist)) * (180.0/math.pi) for size in pixel_size]
    if degrees_per_pixel[0] != degrees_per_pixel[1]:
        print "degrees_per_pixel is incorrect, both values should match.\nThe aspect ratio may not match the screen size. Exiting."
        exit(0)
    pixels_per_degree = round(1.0 / degrees_per_pixel[0]) #partial pixels aren't needed here
    
    gabor_diameter = round(pixels_per_degree)*ecc
    
    linear_spacing = numpy.linspace(-ecc/2.0, ecc/2.0, gabor_diameter + 1)
    [xf, yf] = numpy.meshgrid(linear_spacing, linear_spacing) #equivalent to doing meshgrid(linear_spacing) in matlab
    xf = xf[0:-1, 0:-1]
    yf = yf[0:-1, 0:-1]
    #sigma = 0.5 #width of gaussian
    
    mat_array = numpy.exp((-((xf ** 2) + (yf ** 2))) / (sigma ** 2)) #equation from aaron's code
    gaussian = mat2gray(mat_array)
    
    ramp = numpy.sin(orientation * math.pi/180.0) * xf - numpy.cos(orientation * math.pi / 180.0) * yf
    grating = numpy.sin(2.0 * math.pi * sf * ramp)
    
    g = mat2gray(grating * gaussian)

    return FREQ_DATA._make([pixels_per_degree, gabor_diameter, xf, yf, gaussian, ramp, grating, g])

def modulate_image(gabor_def,
                    visuals,
                    spacials,
                    position,
                    min_contrast=0.0,
                    frequency_data=None):
    
    (pixels_per_degree, gabor_diameter, xf, yf, gaussian, ramp, grating, g) = frequency_data if isinstance(frequency_data, FREQ_DATA) else load_spacial_data(visuals, spacials)
    
    top_left_pos = (position[0] - (gabor_diameter / 2.0), position[1] - (gabor_diameter / 2.0))
    #import time
    #print " "
    #s = time.time()
    patch = gabor_def.rms_matrix[top_left_pos[0] : top_left_pos[0] + gabor_diameter, top_left_pos[1] : top_left_pos[1] + gabor_diameter, :]
    #print "a " + str((time.time() - s) * 1000.0)
    #s = time.time()
    
    patch_avg = gabor_def.avg_matrix[top_left_pos[0] : top_left_pos[0] + gabor_diameter, top_left_pos[1] : top_left_pos[1] + gabor_diameter]
    #patch_avg = numpy.mean(patch, axis=2)
    #print "x = {0}".format(patch_avg)
    #patch_avg = patch[:,:,1]
    #print "b " + str((time.time() - s) * 1000.0)
    #s = time.time()
    R = (patch_avg / 127.0) - 1
    R = R / (numpy.max(numpy.abs(R))) / 2.0
    #print "c " + str((time.time() - s) * 1000.0)
    #s = time.time()
    rms_measure = numpy.std(R + 0.5) / numpy.mean(R + 0.5)
    if min_contrast > 0:
        rms_measure = max(rms_measure, min_contrast)
    #print "d " + str((time.time() - s) * 1000.0)
    g = g * (255.0 * rms_measure)
    g = g - numpy.mean(g)
    
    gabor = numpy.transpose(numpy.tile(g, (3,1,1)), (1,2,0))
    
    return GABOR_DATA._make([top_left_pos, gabor_diameter, gabor_diameter / 2.0, patch, numpy.clip(patch + gabor, 0, 255).astype('uint8')])
    
