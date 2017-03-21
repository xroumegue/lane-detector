import logging
import configparser
from argparse import ArgumentParser, FileType, Action, Namespace
import os

import numpy as np;
import cv2;
import math;


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
    def __init__(self, conf, logger = None):
       self.logger = logger 
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
        self.logger.debug('Vanishing point coordinates: (%.3f, %.3f)', vp[0], vp[1])

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

    def load(self, _file):
        self.img = cv2.imread(_file)

    def compute(self):
        width = self.conf['ipmWidth']
        height = self.conf['ipmHeight']
        right = self.conf['ipmRight']
        left = self.conf['ipmLeft']
        top = self.conf['ipmTop']
        bottom = self.conf['ipmBottom']

        out = np.zeros([height, width, 3], dtype= np.uint8)
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
                        out[y, x] = interpolation(self.img, xPos, yPos)
                    else:
                        # Nearest Neighbour
                        out[y, x] = self.img[yPos.astype(int), xPos.astype(int)]
                else:
                   out[y, x] =  0

        return self.out

class filter:
    """A class filtering road image"""
    def __init__(self, conf, logger = None):
        self.logger = logger
        self.conf = conf

    def customFilter(self, imgIn, lineType= 'vertical'):
        kernel = {
                    'vertical' : {
                        #High pass filter (edge)
                        'x' : np.float32(
                                            [
                                                1.000000e-16, 1.280000e-14, 7.696000e-13,
                                                2.886400e-11, 7.562360e-10, 1.468714e-08,
                                                2.189405e-07, 2.558828e-06, 2.374101e-05,
                                                1.759328e-04, 1.042202e-03, 4.915650e-03,
                                                1.829620e-02, 5.297748e-02, 1.169560e-01,
                                                1.918578e-01, 2.275044e-01, 1.918578e-01,
                                                1.169560e-01, 5.297748e-02, 1.829620e-02,
                                                4.915650e-03, 1.042202e-03, 1.759328e-04,
                                                2.374101e-05, 2.558828e-06, 2.189405e-07,
                                                1.468714e-08, 7.562360e-10, 2.886400e-11,
                                                7.696000e-13, 1.280000e-14, 1.000000e-16
                                            ]
                                            )[:, np.newaxis],
                        # Low pass Filter (smooth)
                        'y' : np.float32(
                                            [
                                                -1.000000e-03, -2.200000e-02, -1.480000e-01,
                                                -1.940000e-01, 7.300000e-01, -1.940000e-01,
                                                -1.480000e-01, -2.200000e-02, -1.000000e-03
                                            ]
                                            )[np.newaxis, :]
                        },
                    'horizontal' : {
                        # High pass filter (edge)
                        'y' : np.float32(
                                            [
                                                -2.384186e-07, -4.768372e-06, -4.482269e-05,
                                                -2.622604e-04, -1.064777e-03, -3.157616e-03,
                                                -6.976128e-03, -1.136112e-02, -1.270652e-02,
                                                -6.776810e-03, 6.776810e-03, 2.156258e-02,
                                                2.803135e-02, 2.156258e-02, 6.776810e-03,
                                                -6.776810e-03, -1.270652e-02, -1.136112e-02,
                                                -6.976128e-03, -3.157616e-03, -1.064777e-03,
                                                -2.622604e-04, -4.482269e-05, -4.768372e-06,
                                                -2.384186e-07
                                            ]
                                            )[:, np.newaxis],
                        # Low pass Filter (smooth)
                        'x' : np.float32(
                                            [
                                                2.384186e-07, 5.245209e-06, 5.507469e-05,
                                                3.671646e-04, 1.744032e-03, 6.278515e-03,
                                                1.778913e-02, 4.066086e-02, 7.623911e-02,
                                                1.185942e-01, 1.541724e-01, 1.681881e-01,
                                                1.541724e-01, 1.185942e-01, 7.623911e-02,
                                                4.066086e-02, 1.778913e-02, 6.278515e-03,
                                                1.744032e-03, 3.671646e-04, 5.507469e-05,
                                                5.245209e-06, 2.384186e-07
                                            ]
                                            )[np.newaxis, :]
                        }
                }
        # filter over the x
        imgNoDC = cv2.subtract(imgIn, cv2.mean(imgIn))
        imgOutx = cv2.filter2D(imgNoDC, -1, kernel[lineType]['x'])
        imgOuty = cv2.filter2D(imgOutx, -1, kernel[lineType]['y'])

        return imgOuty

    def apply(self, imgIn):
        if self.conf['method'] == 'custom':
            imgOut = self.customFilter(imgIn)
        elif self.conf['method'] == 'bilateral':
            imgOut = cv2.bilateralFilter(outImg, 15, 100, 30)
        return imgOut

class laneDetector:
    """A class detecting lanes on road picture"""

    def __init__(self):
        FORMAT = '%(asctime)-15s-%(levelname)-8s-%(message)s'
        self.config = configparser.ConfigParser(inline_comment_prefixes=('#'))
        # create logger
        logging.basicConfig(format=FORMAT)
        self.logger = logging.getLogger('Lane Detection')

    def readConf(self, _file):
        if os.path.isfile(_file):
            self.config.read(_file)
        else:
            self.logging.error("Config file %s does not exist !", _file)

    def filter(self, img):
        conf = {}
        if not self.config.has_section('filter'):
            self.logger.error('No filter configuration found !')
            return

        for (k, v) in self.config.items('filter'):
            conf[k] = v

        self.filter = filter(conf, self.logger)
        imgOut = self.filter.apply(img)
        return imgOut

    def getIPM(self, _file):
        conf = {}
        conf['yaw'] = self.config.getfloat('camera', 'yaw') * np.pi / 180
        conf['pitch'] = self.config.getfloat('camera', 'pitch') * np.pi / 180
        conf['c1'] = np.cos(conf['yaw'])
        conf['s1'] = np.sin(conf['pitch'])
        conf['c2'] = np.cos(conf['yaw'])
        conf['s2'] = np.sin(conf['yaw'])
        conf['fu'] = self.config.getfloat('camera', 'focalLengthX')
        conf['fv'] = self.config.getfloat('camera', 'focalLengthY')
        conf['cu'] = self.config.getfloat('camera', 'opticalCenterX')
        conf['cv'] = self.config.getfloat('camera', 'opticalCenterY')
        conf['h'] = self.config.getfloat('camera', 'cameraHeight')

        conf['ipmWidth']  = self.config.getint('ipm', 'ipmWidth')
        conf['ipmHeight']  = self.config.getint('ipm', 'ipmHeight')
        conf['ipmTop']  = self.config.getint('ipm', 'ipmTop')
        conf['ipmBottom'] = self.config.getint('ipm', 'ipmBottom')
        conf['ipmLeft'] = self.config.getint('ipm', 'ipmLeft')
        conf['ipmRight'] = self.config.getint('ipm', 'ipmRight')
        conf['ipmInterpolation'] = self.config.getint('ipm', 'ipmInterpolation')

        myIpm = ipm(conf, self.logger)

#        cv2.imshow('street', img);
#        cv2.waitKey(0);
#        cv2.destroyAllWindows();
        myIpm.getVanishingPoint()
        self.logger.info('Vanishing point: (%.2f, %.2f)', myIpm.vp[0], myIpm.vp[1])
        myIpm.getROI()
        myIpm.load(_file)
        outImg = myIpm.compute()
        return outImg



def parse_cmdline(parser):
	parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG, default=logging.INFO, help='Be verbose...')
	parser.add_argument('-i', '--image', help='Image file')
	parser.add_argument('-c', '--camera', default='CameraInfo.conf', help='Camera configuration file')
	parser.add_argument('-l', '--lanes', default='Lanes.conf', help='Lane configuration file')

	return parser.parse_args()


def main():
    parser = ArgumentParser(description= "Apply an Inverse Perspective Mapping on a img")
    args = parse_cmdline(parser)

    detector = laneDetector()

    detector.logger.setLevel(args.verbose)

    if not args.image:
        print("I need an image !!")
        parser.print_help()
        return

    detector.logger.debug('Using %s as image', args.image)
    detector.logger.debug('Using %s as camera configuration file', args.camera)
    detector.logger.debug('Using %s as lanes configuration file', args.lanes)

    detector.readConf(args.camera)
    detector.readConf(args.lanes)

    outImg = detector.getIPM(args.image)
    cv2.imwrite("IPM-out.jpg", outImg)
    cv2.imshow('IPM', outImg);
    outImgFiltered = detector.filter(outImg)
    cv2.imshow('IPM Filtered', outImgFiltered);


    cv2.waitKey(0);
    cv2.destroyAllWindows();

if __name__ == '__main__':
    main()
