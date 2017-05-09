#! /usr/bin/env python3

import sys
try:
    import numpy as np
except:
    print("Error while importing numpy, please do: # pip install numpy")
    sys.exit()

try:
    import vispy
    from vispy import app
    from vispy import gloo
except:
    print("Error while importing vispy module, please do: #pip install vispy")
    sys.exit()

try:
    from PIL import Image
except:
    print("Error while importing PIL, please do: #pip install Pillow")
    sys.exit()

from os import listdir
from os.path import isfile, dirname, realpath, join

customLibPath=join(dirname(realpath(__file__)),'../lib/python')
for _dir in listdir(customLibPath):
    sys.path.append(dirname(realpath(join(customLibPath, _dir))))

import laneDetector.ipm as ipm
import logging
import math
from argparse import ArgumentParser, FileType, Action, Namespace
import configparser

FORMAT = '%(asctime)-15s-%(levelname)-5s-%(funcName)-8s-%(lineno)-4s-%(message)s'

FRAGMENT_SHADER_FILENAME = join(dirname(realpath(__file__)), 'ipm.frag')
VERTEX_SHADER_FILENAME = join(dirname(realpath(__file__)), 'ipm.vert')

class IpmCanvas(app.Canvas, ipm):
    def __init__(self, fileName, conf):
        app.Canvas.__init__(self,
                        keys = 'interactive',
                        size = (conf['ipmWidth'], conf['ipmHeight']),
                        position = (0,0),
                        title = 'IPM',
                        show = False,
                        resizable = False)

        ipm.__init__(self, conf, "ipm openGL")

        self._rendertex = gloo.Texture2D(shape=(self.size[1], self.size[0], 4))
        self._fbo = gloo.FrameBuffer(self._rendertex,
                                    gloo.RenderBuffer((self.size[1], self.size[0])))

        try:
            fragmentShaderSourceString = open(FRAGMENT_SHADER_FILENAME).read()
        except:
            self.logger.fatal("%s does not exist !", FRAGMENT_SHADER_FILENAME)
            sys.exit()

        try:
            vertexShaderSourceString = open(VERTEX_SHADER_FILENAME).read()
        except:
            self.logger.fatal("%s does not exist !", VERTEX_SHADER_FILENAME)
            sys.exit()

        self.program = gloo.Program(vertexShaderSourceString, fragmentShaderSourceString)
        self.program["position"] = [(-1, -1), (-1, 1), (1, 1), (-1, -1), (1, 1), (1, -1)]

        img = Image.open(fileName)
        im = np.array(list(img.getdata()),np.uint8).reshape((img.size[1], img.size[0], 3))

        gloo.set_viewport(0, 0, *self.size)

        tex = gloo.Texture2D(im)
        tex.interpolation = 'linear'
        tex.wrapping = 'repeat'
        self.program['iChannel'] = tex
        self.program['iChannelResolution'] = (im.shape[1], im.shape[0], im.shape[2])
        self.program['iResolution'] = (self.size[0], self.size[1], 0.)

        self.getUniforms()
#        self.show()
        self.update()

    def getUniforms(self):
        self._getMi2gMat()
        self._getMg2iMat()
        self._getUniformROI()
        self._getUniformIPM()
        self.program['uH'] = self.conf['h']

    def _getUniformROI(self):
        _ROI = self.getROI()
        uROI = np.array((2,2), dtype=np.float32)
        uROI = [
            (_ROI['x']['min'], _ROI['y']['min']),
            (_ROI['x']['max'], _ROI['y']['max'])
        ]
        self.program['uwROI'] = uROI
        return uROI

    def _getUniformIPM(self):
        uIPM = np.array((2,2), dtype=np.float32)
        uIPM = [
            (self.conf['ipmLeft'], self.conf['ipmBottom']),
            (self.conf['ipmRight'], self.conf['ipmTop'])

        ]
        self.program['uIPM'] = uIPM
        return uIPM

    def _getMi2gMat(self):
        c1 = self.conf['c1']
        s1 = self.conf['s1']
        c2 = self.conf['c2']
        s2 = self.conf['s2']
        fu = self.conf['fu']
        fv = self.conf['fv']
        cu = self.conf['cu']
        cv = self.conf['cv']
        h = self.conf['h']

        Mi2g = np.zeros((4, 4), dtype=np.float32)
        Mi2g = [
                (-c2/fu, s1*s2/fv,       (cu*c2/fu) - (cv*s1*s2/fv) - (c1*s2),   0),
                (s2/fu,  s1*c1/fv,       (-cu*s2/fu) - (cv*s1*c2/fv) - (c1*c2),  0),
                (0,      c1/fv,          (-cv*c1/fv) +s1,                        0),
                (0,      -c1/(h*fv),     (cv*c1/(h*fv)) - (s1/h),                0)
        ]

        self.program['uMi2g'] = Mi2g
        return Mi2g


    def _getMg2iMat(self):
        c1 = self.conf['c1']
        s1 = self.conf['s1']
        c2 = self.conf['c2']
        s2 = self.conf['s2']
        fu = self.conf['fu']
        fv = self.conf['fv']
        cu = self.conf['cu']
        cv = self.conf['cv']
        h = self.conf['h']

        Mg2i = np.zeros((4, 4), dtype=np.float32)

        Mg2i = [
            (fu*c2 + cu*c1*s2,       cu*c1*c2 - s2*fu,       -cu*s1,                 0),
            (s2*(cv*c1 - fv*s1),     c2*(cv*c1 - fv*s1),     -fv*c1 - cv*s1,         0),
            (c1*s2,                  c1*c2,                  -s1,                    0),
            (c1*s2,                  c1*c2,                  -s1,                    0),
        ]

        self.program['uMg2i'] = Mg2i

        return Mg2i

    def on_draw(self, event):
        with self._fbo:
            gloo.clear('black')
            gloo.set_viewport(0, 0, *self.size)
            self.program.draw()
            self.im = self._fbo.read()
            app.quit()

def parse_cmdline(parser):
    parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG, default=logging.INFO, help='Be verbose...')
    parser.add_argument('-i', '--image', help='Image file')
    parser.add_argument('-s', '--show', help='show Image', action='store_const', const=True, default=False)
    parser.add_argument('-o', '--output', help='Output IPM filename', default="ipm.bmp")
    parser.add_argument('-c', '--config', default='ipm.conf', help='IPM configuration file')

    return parser.parse_args()

def main():
    vispy.set_log_level('DEBUG')
    try:
        vispy.use(app='glfw', gl='gl+')
    except RuntimeError as e:
        pass

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

    c = IpmCanvas(args.image, conf)
    app.run()
    ipmImage = c.im
    im = Image.frombuffer("RGBA", (c.im.shape[1], c.im.shape[0]), c.im.copy(order='C'), "raw", "RGBA", 0, 1)
    if args.show is True:
        im.show()
    im.save(args.output)

print("Hit ESC key to quit.")
main()
