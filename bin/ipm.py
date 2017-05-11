#! /usr/bin/env python3
import sys
import logging
try:
    import numpy as np
except:
    print("Error while importing numpy, please do: # pip install numpy")
    sys.exit()

try:
    from PIL import Image
except:
    print("Error while importing PIL, please do: #pip install Pillow")
    sys.exit()


import configparser
from argparse import ArgumentParser, FileType, Action, Namespace
from os import listdir
from os.path import isfile, dirname, realpath, join
customLibPath=join(dirname(realpath(__file__)),'../lib/python')
for _dir in listdir(customLibPath):
    sys.path.append(dirname(realpath(join(customLibPath, _dir))))


import laneDetector
FORMAT = '%(asctime)-15s-%(levelname)-5s-%(funcName)-8s-%(lineno)-4s-%(message)s'

def parse_cmdline(parser):
    parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG, default=logging.INFO, help='Be verbose...')
    parser.add_argument('-i', '--image', help='Image file')
    parser.add_argument('-s', '--show', help='show Image', action='store_const', const=True, default=False)
    parser.add_argument('-o', '--output', help='Output IPM filename', default="ipm.bmp")
    parser.add_argument('-c', '--config', default='ipm.conf', help='IPM configuration file')

    return parser.parse_args()

def main():

    logging.basicConfig(format=FORMAT)
    parser = ArgumentParser(description= "Apply an Inverse Perspective Mapping on a img")
    args = parse_cmdline(parser)
    log = logging.getLogger("ipm openGL")
    log.setLevel(args.verbose)
    log.info("OpenGL acceleration to compute an IPM")

    config = configparser.ConfigParser(inline_comment_prefixes=('#'))
    try:
        config.read(args.config)
    except:
        log.fatal("Incorrect config file!")
        sys.exit()

    conf = {}

    if config.has_section('camera'):
        conf['yaw'] = config.getfloat('camera', 'yaw') * np.pi / 180
        conf['pitch'] = config.getfloat('camera', 'pitch') * np.pi / 180
        conf['c1'] = np.cos(conf['pitch'])
        conf['s1'] = np.sin(conf['pitch'])
        conf['c2'] = np.cos(conf['yaw'])
        conf['s2'] = np.sin(conf['yaw'])
        conf['fu'] = config.getfloat('camera', 'focalLengthX')
        conf['fv'] = config.getfloat('camera', 'focalLengthY')
        conf['cu'] = config.getfloat('camera', 'opticalCenterX')
        conf['cv'] = config.getfloat('camera', 'opticalCenterY')
        conf['h'] = config.getfloat('camera', 'cameraHeight')
    else:
        log.fatal("Configuration file must have a camera section!")
        sys.exit()

    if config.has_section('ipm'):
        conf['ipmWidth']  = config.getint('ipm', 'ipmWidth')
        conf['ipmHeight']  = config.getint('ipm', 'ipmHeight')
        conf['ipmTop']  = config.getint('ipm', 'ipmTop')
        conf['ipmBottom'] = config.getint('ipm', 'ipmBottom')
        conf['ipmLeft'] = config.getint('ipm', 'ipmLeft')
        conf['ipmRight'] = config.getint('ipm', 'ipmRight')
        conf['ipmInterpolation'] = config.getint('ipm', 'ipmInterpolation')
    else:
        log.fatal("Configuration file must have a IPM section!")
        sys.exit()

    c = laneDetector.ipm(conf, "ipm openGL")
    im = c.getIpmFromFile(args.image)
    if args.show is True:
        im.show()
    im.save(args.output)

print("Hit ESC key to quit.")
main()
