/*
  cpymatplotlib.h
*/

#ifndef __CPYMATPLOTLIB_H__
#define __CPYMATPLOTLIB_H__

#ifndef UNICODE
#define UNICODE
#endif

#include <Python.h>
#include <windows.h>
#include <stdio.h>

#ifdef __CPYMATPLOTLIB_MAKE_DLL_
#define __PORT __declspec(dllexport) // make dll mode
#else
#define __PORT __declspec(dllimport) // use dll mode
#endif

typedef unsigned char uchar;
typedef unsigned int uint;

__PORT BOOL APIENTRY DllMain(HINSTANCE, DWORD, LPVOID);
__PORT uint WINAPI testVoid(void);
__PORT uint WINAPI testExport(uint a, uint b);

// must not use WINAPI when returns PyObject *
__PORT PyObject *testPyObject(PyObject *self, PyObject *args, PyObject *kw);
__PORT PyObject *cos_func_np(PyObject *self, PyObject *args);
__PORT PyObject *lissajous_np(PyObject *self, PyObject *args);

#endif // __CPYMATPLOTLIB_H__
