/*
  cpymatplotlib.h
*/

#ifndef __CPYMATPLOTLIB_H__
#define __CPYMATPLOTLIB_H__

#ifndef UNICODE
#define UNICODE
#endif

#include <Python.h>
#include <structmember.h>
#include <frameobject.h>
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
__PORT uint WINAPI cpymVoid(void);
__PORT uint WINAPI cpymExport(uint a, uint b);

#define CPYMATPLOTLIB "cpymatplotlib"

// PyErr_Fetch should be called at the same (stack) layer as MACRO placed on.
// and *MUST* be called PyImport_ImportModule etc *AFTER* PyErr_Fetch
#define CPYMPROCESSEXCEPTION(S) do{ \
  if(PyErr_Occurred()){ \
    PyObject *ptyp, *pval, *ptb; \
    PyErr_Fetch(&ptyp, &pval, &ptb); \
    if(0) fprintf(stderr, "%08x %08x: %s\n", ptb, pval, \
      pval ? PyString_AsString(pval) : "!pval"); \
    PyObject *m = PyImport_ImportModule(CPYMATPLOTLIB); \
    if(!m) fprintf(stderr, "cannot import %s\n", CPYMATPLOTLIB); \
    else{ \
      PyObject *tpl = Py_BuildValue("(s)", S); \
      PyObject *kw = PyDict_New(); \
      if(ptyp) PyDict_SetItemString(kw, "typ", ptyp); \
      if(pval) PyDict_SetItemString(kw, "val", pval); \
      if(ptb) PyDict_SetItemString(kw, "tb", ptb); \
      PyObject_Call(PyObject_GetAttrString(m, "cpymProcessException"), \
        tpl, kw); \
    } \
    PyErr_NormalizeException(&ptyp, &pval, &ptb); \
    PyErr_Clear(); \
    if(0) fprintf(stderr, "cleanup exceptions inside: %s\n", S); \
  } \
}while(0)

// must not use WINAPI when returns PyObject *
__PORT PyObject *cpymProcessException(PyObject *self, PyObject *args, PyObject *kw);
__PORT PyObject *cpymPyObject(PyObject *self, PyObject *args, PyObject *kw);
__PORT PyObject *cpymFunc(PyObject *self, PyObject *args);
__PORT PyObject *cpymFuncKwArgs(PyObject *self, PyObject *args, PyObject *kw);
__PORT PyObject *cpymFuncNoArgs(PyObject *self);
__PORT PyObject *npCos(PyObject *self, PyObject *args);
__PORT PyObject *npLissajous(PyObject *self, PyObject *args);

#endif // __CPYMATPLOTLIB_H__
