#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''test_cpymatplotlib
'''

import sys, os
import time
import ctypes
from ctypes import cast, POINTER, py_object, c_int

import numpy as np
import pylab

sys.path.append('../dll')
import cpymatplotlib

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

def draw_curve(axis, n, th):
  m = n % 4
  ax = axis[m]
  x = np.copy(th)
  y = cpymatplotlib.lissajous_np(x, 4., 3.) # overwrite X-Y
  if m == 0: ax.plot(th, y)
  elif m == 1: ax.plot(x, y)
  elif m == 2: return
  elif m == 3: ax.plot(x, th)
  lines = ax.plot([], [])
  ax.relim()
  ax.grid()

def draw_realtime(seconds):
  pylab.axis([0, 1000, 0, 1])
  pylab.ion() # interactive mode
  pylab.show()
  fig = pylab.figure()
  axis = [fig.add_subplot(221 + _ % 4) for _ in range(4)]
  t = 0
  for i in range(seconds * 10): # about seconds when time.sleep(.01)
    th = np.arange(0, 1.98 * np.pi, 0.05) - t / 20.
    y = cpymatplotlib.cos_func_np(th)
    axis[2].plot(th, y)
    [draw_curve(axis, _, th) for _ in range(4) if _ != 2]
    pylab.draw()
    time.sleep(.01)
    t += 1
    [ax.clear() for ax in axis if ax]
  pylab.ioff()

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

  draw_realtime(20)

if __name__ == '__main__':
  main()
