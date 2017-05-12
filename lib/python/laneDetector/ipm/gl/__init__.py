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


from os import listdir
from os.path import isfile, dirname, realpath, join
import laneDetector

import math


FRAGMENT_SHADER_FILENAME = join(dirname(realpath(__file__)), 'ipm.frag')
VERTEX_SHADER_FILENAME = join(dirname(realpath(__file__)), 'ipm.vert')

class ipmGL(app.Canvas):
    def __init__(self, im, conf, roi, logger):
        vispy.set_log_level('DEBUG')
        try:
            vispy.use(app='glfw', gl='gl+')
        except RuntimeError as e:
            pass

        app.Canvas.__init__(self,
                        keys = 'interactive',
                        size = (conf['ipmWidth'], conf['ipmHeight']),
                        position = (0,0),
                        title = 'IPM',
                        show = False,
                        resizable = False)


        self._rendertex = gloo.Texture2D(shape=(self.size[1], self.size[0], 4))
        self._fbo = gloo.FrameBuffer(self._rendertex,
                                    gloo.RenderBuffer((self.size[1], self.size[0])))

        try:
            fragmentShaderSourceString = open(FRAGMENT_SHADER_FILENAME).read()
        except:
            logger.fatal("%s does not exist !", FRAGMENT_SHADER_FILENAME)
            sys.exit()

        try:
            vertexShaderSourceString = open(VERTEX_SHADER_FILENAME).read()
        except:
            logger.fatal("%s does not exist !", VERTEX_SHADER_FILENAME)
            sys.exit()

        self.program = gloo.Program(vertexShaderSourceString, fragmentShaderSourceString)
        self.program["position"] = [(-1, -1), (-1, 1), (1, 1), (-1, -1), (1, 1), (1, -1)]

        gloo.set_viewport(0, 0, *self.size)

        tex = gloo.Texture2D(im)
        tex.interpolation = 'linear'
        tex.wrapping = 'repeat'
        self.program['iChannel'] = tex
        if len(im.shape) == 3:
            self.program['iChannelResolution'] = (im.shape[1], im.shape[0], im.shape[2])
        else:
            self.program['iChannelResolution'] = (im.shape[1], im.shape[0], 1)
        self.program['iResolution'] = (self.size[0], self.size[1], 0.)

        self.getUniforms(conf, roi)
        self.update()
        app.run()
        return self

    def getUniforms(self, conf, roi):
        self._getMi2gMat(conf)
        self._getMg2iMat(conf)
        self._getUniformROI(conf, roi)
        self._getUniformIPM(conf)
        self.program['uH'] = conf['h']

    def _getUniformROI(self, conf, roi):
        uROI = np.array((2,2), dtype=np.float32)
        uROI = [
            (roi['x']['min'], roi['y']['min']),
            (roi['x']['max'], roi['y']['max'])
        ]
        self.program['uwROI'] = uROI
        return uROI

    def _getUniformIPM(self, conf):
        uIPM = np.array((2,2), dtype=np.float32)
        uIPM = [
            (conf['ipmLeft'], conf['ipmBottom']),
            (conf['ipmRight'], conf['ipmTop'])

        ]
        self.program['uIPM'] = uIPM
        return uIPM

    def _getMi2gMat(self, conf):
        c1 = conf['c1']
        s1 = conf['s1']
        c2 = conf['c2']
        s2 = conf['s2']
        fu = conf['fu']
        fv = conf['fv']
        cu = conf['cu']
        cv = conf['cv']
        h = conf['h']

        Mi2g = np.zeros((4, 4), dtype=np.float32)
        Mi2g = [
                (-c2/fu, s1*s2/fv,       (cu*c2/fu) - (cv*s1*s2/fv) - (c1*s2),   0),
                (s2/fu,  s1*c1/fv,       (-cu*s2/fu) - (cv*s1*c2/fv) - (c1*c2),  0),
                (0,      c1/fv,          (-cv*c1/fv) +s1,                        0),
                (0,      -c1/(h*fv),     (cv*c1/(h*fv)) - (s1/h),                0)
        ]

        self.program['uMi2g'] = Mi2g
        return Mi2g


    def _getMg2iMat(self, conf):
        c1 = conf['c1']
        s1 = conf['s1']
        c2 = conf['c2']
        s2 = conf['s2']
        fu = conf['fu']
        fv = conf['fv']
        cu = conf['cu']
        cv = conf['cv']
        h = conf['h']

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


