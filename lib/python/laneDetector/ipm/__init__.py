import logging
import numpy as np
import math
import laneDetector

def interpolation(array,x,y):
    s = array.shape
    i = math.floor(x)
    j = math.floor(y)
    t = x-i
    u = y-j
    u1 = 1.0-u
    t1 = 1.0-t
    if j==s[0]-1:
        if i==s[1]-1:
            return array[j][i]
        return t*array[j][i]+t1*array[j][i+1]
    if i==s[1]-1:
        return u*array[j][i]+u1*array[j+1][i]
    return t1*u1*array[j][i]+t*u1*array[j+1][i]+t*u*array[j+1][i+1]+t1*u*array[j][i+1]

class ipm:
    def __init__(self, conf):
       self.logger = logging.getLogger(laneDetector.DETECTOR_LOGGER_NAME)
       self.conf = conf
       self.vp = self._getVanishingPoint()
       self.roi = self._getROI()

    def _getVanishingPoint(self):
        c1 = self.conf['c1']
        c2 = self.conf['c2']
        s1 = self.conf['s1']
        s2 = self.conf['s2']
        fu = self.conf['fu']
        fv = self.conf['fv']
        cu = self.conf['cu']
        cv = self.conf['cv']

        vp = np.float32([[ s2/c1, c2/c1, 0 ]]).transpose()

        Tyaw = np.float32([
                            [ c2,   -s2,    0   ],
                            [ s2,   c2,     0   ],
                            [ 0,    0,      1   ],
                        ]
                )

        Tpitch = np.float32([
                                [ 1,    0,      0   ],
                                [ 0,    -s1,    -c1 ],
                                [ 0,    c1,     -s1 ],
                        ]
                )


        T1 = np.float32([
                            [ fu,   0,      cu  ],
                            [ 0,    fv,     cv  ],
                            [ 0,    0,      1   ],
                    ]
                )


        T = np.dot(Tpitch, Tyaw)
        T = T1.dot(T)
        vp = T.dot(vp)
        self.logger.debug('Tyaw:\n%s)', Tyaw)
        self.logger.debug('Tpitch:\n%s)', Tpitch)
        self.logger.debug('T1:\n %s)', T1)
        self.logger.info('Vanishing point coordinates: (%.3f, %.3f)', vp[0], vp[1])

        return vp

    def Tw2i(self, _in):
        c1 = self.conf['c1']
        c2 = self.conf['c2']
        s1 = self.conf['s1']
        s2 = self.conf['s2']
        fu = self.conf['fu']
        fv = self.conf['fv']
        cu = self.conf['cu']
        cv = self.conf['cv']
        h = self.conf['h']

        _in = np.insert(_in, _in.shape[0], [[-h],[1]], axis=0)
        Mw2i = np.float32(
                [
                    [ fu*c2 + cu*c1*s2,	cu*c1*c2 - s2*fu, -cu*s1, 0 ],
                    [ s2*(cv*c1 - fv*s1), c2*(cv*c1 - fv*s1), -fv*c1 - cv*s1, 0 ],
                    [ c1*s2, c1*c2, -s1, 0 ],
                    [ c1*s2, c1*c2, -s1, 0 ],
                ]
            )

        i = np.dot(Mw2i, _in)
        i *= np.ones(i.shape)/i[i.shape[0]-1,:]

        return i[[0,1],:]

    def _getROI(self):
        vp = self.vp

        ipmTop = self.conf['ipmTop']
        ipmBottom = self.conf['ipmBottom']
        ipmRight = self.conf['ipmRight']
        ipmLeft = self.conf['ipmLeft']
        ipmWidth = self.conf['ipmWidth']
        ipmHeight = self.conf['ipmHeight']

        iROI = np.float32(
                [
                    [ vp[0], ipmTop],
                    [ ipmRight, ipmTop ],
                    [ ipmLeft, ipmTop ],
                    [ vp[0], ipmBottom ],
                ]
            ).transpose()

        wROI = self.Ti2w(iROI)

        _ROI = {
                'x' : {
                        'min' : np.amin(wROI[0, :]),
                        'max' : np.amax(wROI[0, :]),
                        'scale' : (np.amax(wROI[0, :]) - np.amin(wROI[0, :]))/ipmWidth,
                    },
                'y' : {
                        'min' : np.amin(wROI[1, :]),
                        'max' : np.amax(wROI[1, :]),
                        'scale' : (np.amax(wROI[1, :]) - np.amin(wROI[1, :]))/ipmHeight,
                    }
                }

        self.logger.debug('wROI: (%.2f, %.2f)(%.2f,%.2f)', _ROI['x']['min'], _ROI['y']['min'], _ROI['x']['max'], _ROI['y']['max'])
        return _ROI


    def Ti2w(self, _in):
        c1 = self.conf['c1']
        c2 = self.conf['c2']
        s1 = self.conf['s1']
        s2 = self.conf['s2']
        fu = self.conf['fu']
        fv = self.conf['fv']
        cu = self.conf['cu']
        cv = self.conf['cv']
        h = self.conf['h']

        _in = np.insert(_in, _in.shape[0], [[1],[1]], axis=0)
        Mi2w = np.float32(
                [
                    [ -c2/fu, s1*s2/fv,	(cu*c2/fu) - (cv*s1*s2/fv) - (c1*s2), 0 ],
                    [ s2/fu, s1*c1/fv, (-cu*s2/fu) - (cv*s1*c2/fv) - (c1*c2), 0 ],
                    [ 0, c1/fv,	(-cv*c1/fv) +s1, 0 ],
                    [ 0, -c1/(h*fv), (cv*c1/(h*fv)) - (s1/h), 0 ],
                ]
            )

        w = np.dot(Mi2w, _in)
        w *= np.ones(w.shape)/w[w.shape[0]-1,:]

        return w[[0,1],:]

    def getVanishingPoint(self):
        return self.vp

    def getROI(self):
        return self.roi

    def compute(self, img):
        width = self.conf['ipmWidth']
        height = self.conf['ipmHeight']
        right = self.conf['ipmRight']
        left = self.conf['ipmLeft']
        top = self.conf['ipmTop']
        bottom = self.conf['ipmBottom']

        out = np.array(np.zeros([height, width]), dtype= img.dtype)
        self.out = out

        wOut = np.zeros([height,width, 2], dtype=np.float32)
        wOut[:,:,0] =  np.repeat(np.linspace(self.roi['x']['min'], self.roi['x']['max'], num=width, dtype= np.float32)[np.newaxis,:], height, 0)
        wOut[:,:,1] =  np.repeat(np.linspace(self.roi['y']['max'], self.roi['y']['min'], num=height, dtype= np.float32)[:,np.newaxis], width, 1)

        wPos = np.reshape(wOut, (width*height, 2));

        iPosVector = self.Tw2i(wPos.transpose()).transpose()
        iPos = np.reshape(iPosVector, (height, width, 2))

        for x in range(width):
            for y in range(height):
                xPos = iPos[y, x, 0]
                yPos = iPos[y, x, 1]

                if left <= xPos <= right and top <= yPos <= bottom:
                    if self.conf['ipmInterpolation'] == 0:
                        #bilinear interpolation
                        out[y, x] = interpolation(img, xPos, yPos)
                    else:
                        # Nearest Neighbour
                        out[y, x] = img[yPos.astype(int), xPos.astype(int)]
                else:
                   out[y, x] =  0

        return self.out

