#based off of GratingThresholdIm function, written in Matlab by Aaron Johnson

import math
import os
import sys
import numpy
import pygame
import time
import argparse
import Tkinter, tkFileDialog
from gabor_util import *

def main():
    parser = argparse.ArgumentParser(description="Test program for displaying gabors on images.")
    parser.add_argument("-save", action="store_true", dest="save", help="Enables saving of the image to the ./output/ folder.")
    parser.add_argument("-nomouse", action="store_true", dest="nomouse", help="If set, the gabor will be centered instead of following the mouse cursor.")
    parser.add_argument("-nobox", action="store_true", dest="nobox", help="If set, the saved output will not be outlined by a box.")
    parser.add_argument("-source", metavar="SourceImage", default="", help="The image to be modulated.")
    parser.add_argument("-localrms", action="store_true", dest="localrms", help="If set, each patch will use the average of its area, otherwise the overall image rms is used.")
    
    parser.add_argument("-no_filter", action="store_true", dest="is_rms", help="If set, the displayed image will not have the RMS contrast filter applied to it.")

    parser.add_argument("-ecc", dest="ecc", default=3, type=int, help="Size of the patch in degrees of visual angle. Defaults to 3.")
    parser.add_argument("-sf", dest="sf", default=2, type=float, help="Spacial frequency of the patch in cycles per degree. Defaults to 0.2.")
    parser.add_argument("-rot", dest="rot", default=45, type=float, help="Orientation of the patch in degrees. Defaults to 45.")
    
    parser.add_argument("-grouprot", dest="grouprot", default=0, type=float, help="Orientation of patch group about the cursor location. Defaults to 0.")
    parser.add_argument("-grouprot_rand", dest="grouprot_rand", action="store_true", help="If set, the patches will be oriented randomly about the cursor.")

    parser.add_argument("-r", "-resolution", dest="resolution", default=[1024, 768], nargs=2, help="Size of the image and screen. Defaults to 1024x768. Image will be resized to fit.")
    parser.add_argument("-s", "-size", dest="size", default=[36.0, 27.0], nargs=2, help="Size of the physical screen in inches. Defaults to 36x27.")
    parser.add_argument("-v", "-vdist", dest="vdist", default=61, help="Distance from the screen of the user. Defaults to 61in.")
    
    parser.add_argument("-f", "-fullscreen", dest="fullscreen", action="store_true", help="Start the program in fullscreen mode.")

    args = parser.parse_args()
    
    if len(args.source) < 1:
        root = Tkinter.Tk()
        root.withdraw()
        args.source = tkFileDialog.askopenfilename(parent=root, initialdir="./", title="Please select an image to modulate.")
        root.destroy()
    
    pygame.init()
    
    resolution = args.resolution
    midpoint = [x/2.0 for x in resolution]
    size = args.size
    vdist = args.vdist
    
    screen = None
    if args.fullscreen:
        screen = pygame.display.set_mode(tuple(resolution), pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode(tuple(resolution), pygame.DOUBLEBUF | pygame.HWSURFACE)
    
    vis_data = VIS_DATA._make([resolution, midpoint, size, vdist])
    
    #negative_multiplier, ori_index = balanceTrials(200, True, ([-1, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    
    if not os.path.exists('output'):
        os.mkdir('output')
    
    spacials = SPATIAL_DATA._make([args.ecc, args.sf, args.rot])
    freqs = load_spacial_data(vis_data, spacials)
    stores = [pygame.Surface((freqs.gabor_diameter, freqs.gabor_diameter)) for i in range(0, 4)]
    
    spread = 100
    if args.grouprot_rand:
        args.grouprot = numpy.random.random() * (math.pi / 4.0)
        print args.grouprot
    else:
        args.grouprot = math.radians(args.grouprot)
    group_formula = (math.sin(args.grouprot), math.cos(args.grouprot))

    store_array = zip(stores, [0, 0, spread, -spread], [-spread, spread, 0, 0])

    clock = pygame.time.Clock()
    gabor_def = load_matrices(args.source, resolution, is_rms=args.is_rms)

    pygame.surfarray.blit_array(screen, gabor_def.rms_matrix)
    pygame.display.flip()
    
    running = True
    while running:
        position = [r / 2 for r in resolution]
        
        if not args.nomouse:
            position = pygame.mouse.get_pos()
            
            position = (
                max(min(position[0], resolution[0] - (freqs.gabor_diameter/2 + spread)), (freqs.gabor_diameter/2 + spread)),
                max(min(position[1], resolution[1] - (freqs.gabor_diameter/2 + spread)), (freqs.gabor_diameter/2 + spread))
            )
        old_gabor_positions = []
        for store, dx, dy in store_array:
            ddx = dx*group_formula[1] - dy*group_formula[0]
            ddy = dx*group_formula[0] + dy*group_formula[1]
            local_pos = (position[0] + ddx, position[1] + ddy)
            gabor = modulate_image(gabor_def, vis_data, spacials, position=local_pos, frequency_data=freqs, use_local_rms=args.localrms)
            pygame.surfarray.blit_array(store, gabor.new_patch)
            screen.blit(store, gabor.position)
            pygame.surfarray.blit_array(store, gabor.old_patch)
            old_gabor_positions.append(gabor.position)
        
        if args.save:
            mat = gabor_def.rms_matrix.copy()
            mat[gabor.position[0]:gabor.position[0] + gabor.size, gabor.position[1]:gabor.position[1] + gabor.size, :] = gabor.new_patch
            image_out = pygame.Surface(resolution)
            pygame.surfarray.blit_array(image_out, mat)
            pygame.image.save(image_out, './output/{4}_ecc{0}-sf{1}-size{2}-rot{3}.jpg'.format(ecc, sf, size_of_gabor, rot, args.source.split('.')[0].split('/')[-1]))
            
        pygame.display.flip()
        
        for (store, dx, dy), pos in zip(store_array, old_gabor_positions):
            screen.blit(store, pos)
        
        clock.tick()
        sys.stdout.write("\rfps = " + str(clock.get_fps()))
        sys.stdout.flush()
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                    running = False
            elif evt.type == pygame.KEYDOWN:
                    pressed_key = evt.key
                    
                    if pressed_key == pygame.K_ESCAPE:
                        running = False
                    if pressed_key == pygame.K_SPACE:
                        running = False
                    if pressed_key == pygame.K_r:
                        args.rot += 5
                        spacials = SPATIAL_DATA._make([args.ecc, args.sf, args.rot])
                        freqs = load_spacial_data(vis_data, spacials)
                    if pressed_key == pygame.K_f:
                        args.rot -= 5
                        spacials = SPATIAL_DATA._make([args.ecc, args.sf, args.rot])
                        freqs = load_spacial_data(vis_data, spacials)
                       
if __name__ == "__main__":
    main()
    print ""
