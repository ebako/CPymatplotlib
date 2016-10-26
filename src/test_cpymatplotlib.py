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
testVoid_cdll = ctypes.cdll.LoadLibrary(dllname).testVoid
testExport_cdll = ctypes.cdll.LoadLibrary(dllname).testExport

# testPyObject = ctypes.cdll.LoadLibrary(dllname).testPyObject # BAD
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
  print 'result0: ', testVoid_cdll()
  # print 'result2: ', testExport_cdll(3, 4)
  ## ValueError: Procedure called with not enough arguments (8 bytes missing)
  ##             or wrong calling convention

  '''
  # old: test code for 'return po;' before link python25.dll
  # old: cannot get PyObject as cast(returned value) ?
  print 'resultPO imm: ', testPyObject(511)

  print 'resultPO p_o: ', testPyObject(py_object(c_int(511)))

  p = cast(testPyObject(py_object((c_int * 2)(255, 511))), POINTER(c_int))
  print 'resultPO pcls: ', p
  print 'resultPO cls: ', p[0], p[1]

  p = cast(testPyObject(py_object(TestA())), POINTER(py_object))
  print 'resultPO pTA: ', p
  print 'resultPO TA: ', p.a, p.b
  '''

  # new: test code (rename python.exe -> python25.dll and create implib)
  #      and link to .DLL but failure with generated 'NOT a python module' ?
  #  It may be same problem that without py_object() in ctypes causes failure
  #  or PyArg_ParseTuple() will failure too ?
  #  Why that problem will be solved when it create as .pyd
  #  and load WINAPI functions by .LoadLibrary(dllname) ?

  if False: # WindowsError: exception: access violation reading 0x00000203
    p = testPyObject(511)
    print 'resultPO: ', p

  if False: # WindowsError: exception: access violation reading 0x00000028
    p = testPyObject(py_object(511))
    print 'resultPO: ', p

  if False: # ctypes.ArgumentError: argument 1:
    # <type 'exceptions.TypeError'>: Don't know how to convert parameter 1
    p = testPyObject((511, 255.0, 'teststring'))
    print 'resultPO: ', p

  if False: # WindowsError: exception: access violation reading 0x00000028
    p = testPyObject(py_object((511, 255.0, 'teststring')))
    print 'resultPO: ', p

  if False: # WindowsError: exception: access violation reading 0x00000054
    p = testPyObject(i=511, d=255.0, s='teststring')
    print 'resultPO: ', p

  if False: # WindowsError: exception: access violation reading 0x00000054
    p = testPyObject(**{'i': 511, 'd': 255.0, 's': 'teststring'})
    print 'resultPO: ', p

  if False: # ctypes.ArgumentError: argument 1:
    # <type 'exceptions.TypeError'>: Don't know how to convert parameter 1
    p = testPyObject({'i': 511, 'd': 255.0, 's': 'teststring'})
    print 'resultPO: ', p

  if False: # WindowsError: exception: access violation reading 0x00000028
    p = testPyObject(py_object({'i': 511, 'd': 255.0, 's': 'teststring'}))
    print 'resultPO: ', p

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
