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
cpymVoid = dllm.cpymVoid
cpymExport = dllm.cpymExport

cpymPyObject = cpymatplotlib.cpymPyObject # or 'from cpymatplotlib import *'

NAXIS = 4

def draw_curve(axis, n, th):
  m = n % NAXIS
  ax = axis[m]
  x = np.copy(th)
  y = cpymatplotlib.npLissajous(x, 4., 3.) # overwrite X-Y
  if m == 0: ax.plot(th, y)
  elif m == 1: ax.plot(x, y)
  elif m == 2: return
  elif m == 3: ax.plot(x, th)
  lines = ax.plot([], [])
  ax.relim()
  ax.grid()

def draw_realtime(seconds):
  # pylab.axis([0, 1000, 0, 1])
  pylab.ion() # interactive mode
  fig = pylab.figure()
  canvas = fig.canvas
  tkc = canvas.get_tk_widget()
  axis = [fig.add_subplot(221 + _ % NAXIS) for _ in range(NAXIS)]
  tkc.bind('<Destroy>', lambda e: e.widget.after_cancel(tid), '+')
  def incnum(t):
    if t >= seconds * 10: return # about seconds when .after(10, ...)
    [ax.clear() for ax in axis if ax]
    th = np.arange(0, 1.98 * np.pi, 0.05) - t / 20.
    y = cpymatplotlib.npCos(th)
    axis[2].plot(th, y)
    [draw_curve(axis, _, th) for _ in range(NAXIS) if _ != 2]
    canvas.draw()
    global tid
    tid = tkc.after(10, incnum, t + 1)
  incnum(0)
  pylab.show()
  pylab.close('all')
  pylab.ioff()

def test_funcs():
  print 'in'

  print 'result0: ', cpymVoid()
  print 'result2: ', cpymExport(3, 4)

  help(cpymatplotlib)

  p = cpymPyObject(511, 255.0, 'teststring')
  print 'resultPO: [%s]' % str(p)

  # o = cpymatplotlib.Nobject(a=456, b=123, c='enroute')
  o = cpymatplotlib.Cbject(a=456, b=123, c='enroute')
  p = cpymPyObject(i=511, d=255.0, s='teststring', a=o)
  print 'resultPO: [%s]' % str(p)

  p = cpymatplotlib.cpymFuncNoArgs()
  print 'resultPO: [%s]' % str(p)

  p = cpymatplotlib.cpymFunc(511, 255.0, 'teststring')
  print 'resultPO: [%s]' % str(p)

  # o = cpymatplotlib.Nobject(7, 5, 'xyz')
  o = cpymatplotlib.Cbject(); o.a=7; o.c='xyz' # test AttributeError
  try:
    p = cpymatplotlib.cpymFuncKwArgs(i=511, d=255.0, s='teststring', a=o)
  except (Exception, AttributeError, ), e: # not caught anymore
    print '**** err [%s] ****' % str(e)
  print 'resultPO: [%s]' % str(p)

  print 'out'

def main():
  test_funcs()
  draw_realtime(20)

if __name__ == '__main__':
  main()
  print 'done.'
