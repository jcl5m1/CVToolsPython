#Copyright [2014] [Google]
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

__author__ = 'johnnylee'
__version__ = '2014.10.29'

import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', action='store', dest='dirname', help='directory of PLY files (binary little endian or ascii)')
    parser.add_argument('-x', dest='xOffset', help='Horz offset for corner split', default=0)
    parser.add_argument('-corner', action="store_true", default=False, help='Corner Dataset')
    parser.add_argument('-rotz', action="store_true", default=False, help='Rotate 90 along Z axis')
    parser.add_argument('-best_fit', action="store_true", default=False, help='Use Best Fit Plane')
    results = parser.parse_args()

if os.path.isdir(results.dirname):
    for r, d, f in os.walk(results.dirname):
        for file in f:
            if str(".ply") in file:
                ply = r+'/'+file
                cmd = "python plyDepthTester.py -f " + ply + " -csv"
                if results.corner:
                    cmd += " -corner"
                if results.best_fit:
                    cmd += " -best_fit"
                if results.rotz:
                    cmd += " -rotz"

                os.system(cmd)