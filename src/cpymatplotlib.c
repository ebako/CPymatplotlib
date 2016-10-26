/*
  cpymatplotlib.c

  ## needless cpymatplotlib.def (auto generated and --add-stdcall-alias)

  gcc -m32 -shared -o cpymatplotlib.pyd cpymatplotlib.c \
    -I/python25/lib/site-packages/numpy/core/include \
    -I/python25/lib/site-packages/numpy/numarray \
    -I/python25/include \
    -I../include \
    -L/python25/libs -lpython25 \
    -Wl,-mi386pe,--add-stdcall-alias\
      ,--out-implib=cpymatplotlib.lib,--output-def=cpymatplotlib.def
    # ,--add-cdecl-alias # unrecognized option
    # ,--cref

  test_cpymatplotlib.py

  Example of wrapping the cos function from math.h using the Numpy-C-API.
  http://www.turbare.net/transl/scipy-lecture-notes/advanced/interfacing_with_c/interfacing_with_c.html
    2.8. C interface (2.8.2. Python-C-Api) (2.8.2.2. Numpy)
    http://www.turbare.net/transl/scipy-lecture-notes/advanced/interfacing_with_c/interfacing_with_c.html#id1

  numpy.get_include()
  -I/.../lib/site-packages/numpy/core/include
    ( lib/site-packages/numpy/core/include/numpy/*.h )
  -I/.../lib/site-packages/numpy/numarray
    ( lib/site-packages/numpy/numarray/numpy/*.h )
  -I/.../include
    ( .../include/Python.h )
  -L.../libs -lpython*
    ( .../libs/python*.lib )
  ## needless _fake_python25.def
  # -L. -l_fake_python*
  #   ( ./_fake_python*.lib )
  # dlltool -mi386 --dllname python*.dll \
  #   --input-def _fake_python*.def --output-lib lib_fake_python*.a
*/

#define __CPYMATPLOTLIB_MAKE_DLL_
#include <cpymatplotlib.h>

#include <numpy/arrayobject.h>
#include <math.h>

#define DEBUGLOG 0
#define TESTLOG "_test_dll_.log"

__PORT uint WINAPI testVoid(void)
{
  FILE *fp = fopen(TESTLOG, "ab");
  fprintf(fp, "testVoid\n");
  fclose(fp);
  return 0;
}

__PORT uint WINAPI testExport(uint a, uint b)
{
  FILE *fp = fopen(TESTLOG, "ab");
  fprintf(fp, "testExport %d %d\n", a, b);
  fclose(fp);
  return a + b;
}

__PORT PyObject *testPyObject(PyObject *self, PyObject *args, PyObject *kw)
{
  int i;
  double d;
  char *s;
  PyObject *a = NULL;
  PyObject *pdi = PyDict_New();

  // PyObject *pm_ctypes = PyImport_ImportModule("ctypes");
  // PyObject *pm_cv = PyImport_ImportModule("opencv.cv");
  // PyObject *pm__cv = PyImport_ImportModule("opencv._cv");

  FILE *fp = fopen(TESTLOG, "ab");
  fprintf(fp, "testPyObject %08x %08x %08x\n",
    (uchar *)self, (uchar *)args, (uchar *)kw);
  fclose(fp);

  // if(obj == Py_None){ }

  char *keys[] = {"i", "d", "s", "a", NULL};
  if(!PyArg_ParseTupleAndKeywords(args, kw, "|idsO", keys, &i, &d, &s, &a)){
    FILE *fp = fopen(TESTLOG, "ab");
    fprintf(fp, "ERROR: PyArg_ParseTupleAndKeywords()\n");
    fclose(fp);
    return NULL;
  }else{
    FILE *fp = fopen(TESTLOG, "ab");
    fprintf(fp, "SUCCESS: PyArg_ParseTupleAndKeywords(%d, %23.17f, %s, %08x)\n", i, d, s, (char *)a);
    fclose(fp);
  }

  if(a){
    char *ks[] = {"a", "b", "c"};
    int i;
    for(i = 0; i < sizeof(ks) / sizeof(ks[0]); ++i)
      PyDict_SetItemString(pdi, ks[i], PyObject_GetAttrString(a, ks[i]));
  }

  // return self; // wrong way
  // return NULL; // ValueError: NULL pointer access
  // Py_RETURN_NONE; // __Py_NoneStruct
  // return Py_BuildValue("O", self); // _Py_BuildValue (INCREFs)
  // return Py_BuildValue("N", self); // _Py_BuildValue (steals a reference)
  return Py_BuildValue("{iisisdsssO}", 5, 2, "i", i, "d", d, "s", s, "o", pdi);
}

__PORT PyObject *cos_func_np(PyObject *self, PyObject *args)
{
  PyArrayObject *in_array;
  PyObject *out_array;
#if 0 // NpyIter and PyArray_NewLikeArray() is available for new version
  NpyIter *in_iter;
  NpyIter *out_iter;
  NpyIter_IterNextFunc *in_iternext;
  NpyIter_IterNextFunc *out_iternext;
#endif

  /* parse single numpy array argument */
  if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array)) return NULL;
#if 0 // NpyIter and PyArray_NewLikeArray() is available for new version
  /* construct the output array, like the input array */
  out_array = PyArray_NewLikeArray(in_array, NPY_ANYORDER, NULL, 0);
  if(out_array == NULL) return NULL;
  /* create the iterators */
  in_iter = NpyIter_New(in_array, NPY_ITER_READONLY, NPY_KEEPORDER,
    NPY_NO_CASTING, NULL);
  if(in_iter == NULL) goto fail;
  out_iter = NpyIter_New((PyArrayObject *)out_array, NPY_ITER_READWRITE,
    NPY_KEEPORDER, NPY_NO_CASTING, NULL);
  if(out_iter == NULL){ NpyIter_Deallocate(in_iter); goto fail; }
  in_iternext = NpyIter_GetIterNext(in_iter, NULL);
  out_iternext = NpyIter_GetIterNext(out_iter, NULL);
  if(in_iternext == NULL || out_iternext == NULL){
    NpyIter_Deallocate(in_iter);
    NpyIter_Deallocate(out_iter);
    goto fail;
  }
  double **in_dataptr = (double **)NpyIter_GetDataPtrArray(in_iter);
  double **out_dataptr = (double **)NpyIter_GetDataPtrArray(out_iter);
  /* iterate over the arrays */
  do{
    **out_dataptr = cos(**in_dataptr);
  }while(in_iternext(in_iter) && out_iternext(out_iter));
  /* clean up and return the result */
  NpyIter_Deallocate(in_iter);
  NpyIter_Deallocate(out_iter);
#else
  /* construct the output array */
#if 1 // without initialize
  out_array = PyArray_SimpleNew(
    PyArray_NDIM(in_array), PyArray_DIMS(in_array), PyArray_TYPE(in_array));
#else // reference (not copied data)
  out_array = PyArray_SimpleNewFromData(
    PyArray_NDIM(in_array), PyArray_DIMS(in_array), PyArray_TYPE(in_array),
    PyArray_DATA(in_array)); // reference (not copied data) Py_INCREF ?
#endif
  if(out_array == NULL) return NULL;
  else{
    double *din = (double *)PyArray_DATA(in_array);
    double *dout = (double *)PyArray_DATA(out_array);
    int i;
#if DEBUGLOG
    FILE *fp = fopen(TESTLOG, "ab");
    fprintf(fp, "NDIM: %d\n", PyArray_NDIM(in_array));
    fprintf(fp, "SHAPE: %d\n", PyArray_DIMS(in_array)[0]);
    fclose(fp);
#endif
    // overwrite same area when reference and memory will be broken
    for(i = 0; i < PyArray_DIMS(in_array)[0]; ++i) *dout++ = cos(*din++);
  }
#endif
  Py_INCREF(out_array);
  return out_array;

fail:
  Py_XDECREF(out_array);
  return NULL;
}

static PyMethodDef cpymatplotlib_methods[] = {
  {"testPyObject", (PyCFunction)testPyObject,
    METH_VARARGS | METH_KEYWORDS, "*args, **kw:\n"
    " i:\n"
    " d:\n"
    " s:\n"
    " a:\n"
    "result: dict"},
  {"cos_func_np", (PyCFunction)cos_func_np, METH_VARARGS,
    "evaluate the cosine on a numpy array"},
  {NULL, NULL, 0, NULL}
};

static char cpymatplotlib_docstr[] = \
  "about this module\ntestdocstr\ntestdocstr\ntestdocstr";

PyMODINIT_FUNC initcpymatplotlib()
{
  Py_InitModule3("cpymatplotlib", cpymatplotlib_methods, cpymatplotlib_docstr);
  /* IMPORTANT: this must be called */
  import_array();
}

BOOL APIENTRY DllMain(HINSTANCE inst, DWORD reason, LPVOID reserved)
{
  switch(reason){
  case DLL_PROCESS_ATTACH:
    break;
  case DLL_PROCESS_DETACH:
    break;
  case DLL_THREAD_ATTACH:
    break;
  case DLL_THREAD_DETACH:
    break;
  default:
    break;
  }
  return TRUE;
}
