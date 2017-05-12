#! /usr/bin/env python3
import sys
from os import listdir
from os.path import isfile, dirname, realpath, join
customLibPath=join(dirname(realpath(__file__)),'../lib/python')
for _dir in listdir(customLibPath):
    sys.path.append(dirname(realpath(join(customLibPath, _dir))))

from laneDetector.ipm.gl import main
print("Hit ESC key to quit.")
main()
