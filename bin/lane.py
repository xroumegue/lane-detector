from argparse import ArgumentParser, FileType, Action, Namespace
from os import listdir
from os.path import isfile, dirname, realpath, join
from sys import path
import logging
import cv2

customLibPath=join(dirname(realpath(__file__)),'../lib/python')
for _dir in listdir(customLibPath):
    path.append(dirname(realpath(join(customLibPath, _dir))))


import laneDetector

def parse_cmdline(parser):
	parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG, default=logging.INFO, help='Be verbose...')
	parser.add_argument('-i', '--image', help='Image file')
	parser.add_argument('-c', '--camera', default='CameraInfo.conf', help='Camera configuration file')
	parser.add_argument('-l', '--lanes', default='Lanes.conf', help='Lane configuration file')

	return parser.parse_args()

def main():
    parser = ArgumentParser(description= "Apply an Inverse Perspective Mapping on a img")
    args = parse_cmdline(parser)

    detector = laneDetector.laneDetector()

    detector.logger.setLevel(args.verbose)

    if not args.image or not isfile(args.image):
        parser.print_help()
        return

    detector.logger.debug('Using %s as image', args.image)
    detector.logger.debug('Using %s as camera configuration file', args.camera)
    detector.logger.debug('Using %s as lanes configuration file', args.lanes)

    detector.readConf(args.camera)
    detector.readConf(args.lanes)

    # Load Image
    detector.load(args.image)
    detector.showImage('original', detector.rawImage)
    # IPM
    outImg = detector.getIPM(useRaw = False)
#    cv2.imwrite("IPM-out.jpg", outImg)
    detector.showImage('IPM', outImg)

    # Filter out "noise"
    outImgFiltered = detector.filter(outImg)
    detector.showImage('IPM Filtered', outImgFiltered);

    # Threshold
    outImgThresholded = detector.threshold(outImg)
    detector.showImage('IPM Thresholded', outImgThresholded);

    # Detecting lines
    myLines = detector.lines(outImgThresholded)
    for _line in myLines:
        __line = _line.getCartesian()
        cv2.line(outImgThresholded, tuple(round(float(_)) for _ in __line[0]), tuple(round(float(_)) for _ in __line[1]), (255, 0, 0), 1)

    detector.showImage('IPM vertical lines', outImgThresholded);
    cv2.waitKey(0);
    cv2.destroyAllWindows();

    detector.ransac(outImgThresholded, myLines)

if __name__ == '__main__':
    main()
