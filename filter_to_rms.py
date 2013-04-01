import gabor_util
import numpy
import os
from PIL import Image
import argparse
    
parser = argparse.ArgumentParser(description="Converts all images in the source directory to rms contrasted versions in the destination directory.")
parser.add_argument("-s", "--src", default="./image_set/", help="Source directory of images.", dest="source")
parser.add_argument("-d", "--dest", default="./rms_image_set/", help="Destination directory to write images to.", dest="dest")
parser.add_argument("-c", "--contrast", default=0.2, type=float, help="RMS contrast value to apply to the image, defaults to 0.2", dest="contrast")
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true")

args = parser.parse_args()

if os.path.exists(args.dest) == False:
    try:
        os.mkdir(args.dest)
    except:
        print "Could not create destination directory '%s'.".format(args.dest)
        exit()
if os.path.exists(args.source) == False:
    print "Source directory '%s' does not exist.".format(args.source)
    exit()

for filename in os.listdir("./image_set"):
    if ".jpg" not in filename:
        continue
    in_filename = "{0}/{1}".format(args.source, filename)
    out_filename = "{0}/{1}".format(args.dest, filename)
    
    if args.verbose:
        print "Converting {0} into {1}".format(os.path.join(args.source, filename), os.path.join(args.dest, filename))

    orig = Image.open(in_filename)
    as_mat = numpy.array(orig)
    rms = gabor_util.color_rms(as_mat, args.contrast)
    out = Image.fromarray(rms)
    out.save(out_filename)
