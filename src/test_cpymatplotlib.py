#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''test_cpymatplotlib
'''

import sys, os

import cpymatplotlib
import numpy as np
import pylab

import ctypes
from ctypes import cast, POINTER, py_object, c_int

# tdll = ctypes.cdll
tdll = ctypes.windll
dllname = 'cpymatplotlib.pyd'
dllm = tdll.LoadLibrary(dllname)
testVoid = dllm.testVoid
testExport = dllm.testExport

testPyObject = cpymatplotlib.testPyObject # or 'from cpymatplotlib import *'

class TestA(object):
  def __init__(self, a, b, c):
    self.a = a
    self.b = b
    self.c = c

def main():
  print 'in'

  print 'result0: ', testVoid()
  print 'result2: ', testExport(3, 4)

  help(cpymatplotlib)

  p = testPyObject(511, 255.0, 'teststring')
  print 'resultPO: ', p

  o = TestA(456, 123, 'enroute')
  p = testPyObject(i=511, d=255.0, s='teststring', a=o)
  print 'resultPO: ', p

  print 'out'

  x = np.arange(0, 2 * np.pi, 0.1)
  y = cpymatplotlib.cos_func_np(x)
  pylab.plot(x, y)
  pylab.show()

if __name__ == '__main__':
  main()
