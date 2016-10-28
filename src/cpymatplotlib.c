/*
  cpymatplotlib.c

  >mingw32-make -f makefile.tdmgcc64
  >test_cpymatplotlib.py

  Example of wrapping the cos function from math.h using the Numpy-C-API.
  http://www.turbare.net/transl/scipy-lecture-notes/advanced/interfacing_with_c/interfacing_with_c.html
    2.8. C interface (2.8.2. Python-C-Api) (2.8.2.2. Numpy)
    http://www.turbare.net/transl/scipy-lecture-notes/advanced/interfacing_with_c/interfacing_with_c.html#id1
*/

#define __CPYMATPLOTLIB_MAKE_DLL_
#include <cpymatplotlib.h>

#include <numpy/arrayobject.h>
#include <math.h>

#define DEBUGLOG 0
#define TESTLOG "../dll/_test_dll_.log"

__PORT uint WINAPI cpymVoid(void)
{
  FILE *fp = fopen(TESTLOG, "ab");
  fprintf(fp, "cpymVoid\n");
  fclose(fp);
  return 0;
}

__PORT uint WINAPI cpymExport(uint a, uint b)
{
  FILE *fp = fopen(TESTLOG, "ab");
  fprintf(fp, "cpymExport %d %d\n", a, b);
  fclose(fp);
  return a + b;
}

__PORT PyObject *cpymPyObject(PyObject *self, PyObject *args, PyObject *kw)
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
  fprintf(fp, "cpymPyObject %08x %08x %08x\n",
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

__PORT PyObject *cpymFunc(PyObject *self, PyObject *args)
{
  int i;
  double d;
  char *s;

  FILE *fp = fopen(TESTLOG, "ab");
  fprintf(fp, "cpymFunc %08x %08x\n", (uchar *)self, (uchar *)args);
  fclose(fp);

  // if(obj == Py_None){ }

  if(!PyArg_ParseTuple(args, "ids", &i, &d, &s)){
    FILE *fp = fopen(TESTLOG, "ab");
    fprintf(fp, "ERROR: PyArg_ParseTuple()\n");
    fclose(fp);
    return NULL;
  }else{
    FILE *fp = fopen(TESTLOG, "ab");
    fprintf(fp, "SUCCESS: PyArg_ParseTuple(%d, %23.17f, %s)\n", i, d, s);
    fclose(fp);
  }

  // return self; // wrong way ?
  // return NULL; // ValueError: NULL pointer access
  // Py_RETURN_NONE; // __Py_NoneStruct
  // return Py_BuildValue("O", self); // _Py_BuildValue (INCREFs)
  // return Py_BuildValue("N", self); // _Py_BuildValue (steals a reference)
  return Py_BuildValue("{iisisdss}", 9, 8, "i", i, "d", d, "s", s);
}

__PORT PyObject *cpymFuncKwArgs(PyObject *self, PyObject *args, PyObject *kw)
{
  int i;
  double d;
  char *s;
  PyObject *a = NULL;
  PyObject *pdi = PyDict_New();

  FILE *fp = fopen(TESTLOG, "ab");
  fprintf(fp, "cpymFuncKwArgs %08x %08x %08x\n",
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

  // return self; // wrong way ?
  // return NULL; // ValueError: NULL pointer access
  // Py_RETURN_NONE; // __Py_NoneStruct
  // return Py_BuildValue("O", self); // _Py_BuildValue (INCREFs)
  // return Py_BuildValue("N", self); // _Py_BuildValue (steals a reference)
  return Py_BuildValue("{iisisdsssO}", 0, 3, "i", i, "d", d, "s", s, "o", pdi);
}

__PORT PyObject *cpymFuncNoArgs(PyObject *self)
{
  char *rs[] = {"0: hello", "1: cpymatplotlib", "2: zzzzzzzz"};
  PyObject *pls = PyList_New(sizeof(rs) / sizeof(rs[0])); // (PyListObject *)
  PyObject *pm_hashlib = PyImport_ImportModule("hashlib");

  FILE *fp = fopen(TESTLOG, "ab");
  fprintf(fp, "cpymFuncNoArgs %08x\n", (uchar *)self);
  fclose(fp);

  // return self; // wrong way ?
  // return NULL; // ValueError: NULL pointer access
  // Py_RETURN_NONE; // __Py_NoneStruct
  // return Py_BuildValue("O", self); // _Py_BuildValue (INCREFs)
  // return Py_BuildValue("N", self); // _Py_BuildValue (steals a reference)
  if(pls){
    int i;
    for(i = 0; i < sizeof(rs) / sizeof(rs[0]); ++i)
      PyList_SetItem(pls, i, Py_BuildValue("s", rs[i]));
    PyList_Append(pls, PyInt_FromLong(9L)); // may use it when PyList_New(0);
    PyList_Append(pls, PyInt_FromLong(1L << 31)); // 0x80000000 -> -2147483648
    /***************************************************************
    * raise warning: left shift count >= width of type [-Wshift-count-overflow]
    ***************************************************************/
    PyList_Append(pls, PyInt_FromLong(1L << 32)); // 0x100000000 circle -> 0
    PyList_Append(pls, PyInt_FromLong((1L << 32) - 1)); // 0xFFFFFFFF -> -1
    PyList_Append(pls, PyInt_FromLong((1L << 32) - 2147483648)); // -2147483648
    PyList_Append(pls, PyInt_FromLong((1L << 32) - 2147483649)); // 2147483647
    PyList_Append(pls, PyString_FromStringAndSize("leaks-size-align4...", 40));
    PyList_Append(pls, PyString_FromString("fromstring"));
    // not support %f %lf on Python 2.5 ?
    PyList_Append(pls, PyString_FromFormat("(%d, %23.17f, %lf)", 7, 2.5, 6.0));
    {
      char fm[256];
      sprintf(fm, "(%23.17f, %lf)", 2.5, 6.0);
      PyList_Append(pls, PyString_FromFormat("(%d, %-s)", 7, fm));
    }
    // PyList_Append(pls, PyString_FromFormatV("", varargs));
#if 1
    { // *** BUG ? (GC) *** http://docs.python.jp/2/extending/extending.html
      FILE *fp = fopen(TESTLOG, "ab");
      fprintf(fp, "-0\n"); fflush(fp);
      PyObject *item = PyList_GetItem(pls, 0);
      fprintf(fp, "-1: %08x\n", item); fflush(fp);
      Py_INCREF(item); // block for __del__()
      fprintf(fp, "-2\n"); fflush(fp);
      PyList_SetItem(pls, 0, PyInt_FromLong(0L)); // replace [0] and __del__()
      fprintf(fp, "-3\n"); fflush(fp);
      // PyObject_Print(item, fp, Py_PRINT_RAW); // emmit UAE ?
      fprintf(fp, "[%s]\n", PyString_AsString(PyObject_Repr(item)));
      fprintf(fp, "[%s]\n", PyString_AsString(PyObject_Repr(pls)));
      fprintf(fp, "-4\n"); fflush(fp);
      Py_DECREF(item); // release for __del__()
      fprintf(fp, "-5\n"); fflush(fp);
      fclose(fp);
    }
#endif
    return Py_BuildValue("O", pls);
  }
  return Py_BuildValue("s", "hello cpymatplotlib TEST");
}

__PORT PyObject *npCos(PyObject *self, PyObject *args)
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

__PORT PyObject *npLissajous(PyObject *self, PyObject *args)
{
  PyArrayObject *in_array;
  PyObject *out_array;
  double rs = 1.0, rc = 1.0;

  if(!PyArg_ParseTuple(args, "O!dd", &PyArray_Type, &in_array, &rs, &rc))
    return NULL;
  out_array = PyArray_SimpleNew(
    PyArray_NDIM(in_array), PyArray_DIMS(in_array), PyArray_TYPE(in_array));
  if(out_array == NULL) return NULL;
  else{
    double *din = (double *)PyArray_DATA(in_array);
    double *dout = (double *)PyArray_DATA(out_array);
    double th;
    int i;
    for(i = 0; i < PyArray_DIMS(in_array)[0]; ++i){ // overwrite X-Y
      th = *din;
      *dout++ = sin(th * rs);
      *din++ = cos(th * rc);
    }
  }
  Py_INCREF(out_array);
  return out_array;

fail:
  Py_XDECREF(out_array);
  return NULL;
}

typedef struct _Noddy {
  PyObject_HEAD
  PyObject *first;
  PyObject *last;
  int number;
} Noddy;

static int Noddy_traverse(Noddy *self, visitproc visit, void *arg)
{
  Py_VISIT(self->first);
  Py_VISIT(self->last);
  return 0;
}

static int Noddy_clear(Noddy *self)
{
  Py_CLEAR(self->first);
  Py_CLEAR(self->last);
  return 0;
}

static void Noddy_dealloc(Noddy *self)
{
  Noddy_clear(self);
  self->ob_type->tp_free((PyObject *)self);
}

static PyObject *Noddy_new(PyTypeObject *typ, PyObject *args, PyObject *kw)
{
  // may call typ->tp_base->tp_new(...);
  // or PyObject_Call(PyObject_GetAttrString(super, "__new__"), args, kw);
  Noddy *self = (Noddy *)typ->tp_alloc(typ, 0);
  if(!self) return NULL;
  self->first = PyString_FromString("");
  if(!self->first){ Py_DECREF(self); return NULL; }
  self->last = PyString_FromString("");
  if(!self->last){ Py_DECREF(self); return NULL; }
  self->number = 0;
  return (PyObject *)self;
}

static int Noddy_init(Noddy *self, PyObject *args, PyObject *kw)
{
  // may call typ->tp_base->tp_init(...);
  // or PyObject_Call(PyObject_GetAttrString(super, "__init__"), args, kw);
  // but with self ?
  static char *ks[] = {"first", "last", "number", NULL};
  PyObject *first = NULL, *last = NULL, *tmp;
  if(!PyArg_ParseTupleAndKeywords(args, kw, "|SSi", ks, &first, &last, &self->number))
    return -1;
  if(first){
    tmp = self->first;
    Py_INCREF(first);
    self->first = first;
    Py_DECREF(tmp); // must *NOT* decrement reference counter before assign
  }
  if(last){
    tmp = self->last;
    Py_INCREF(last);
    self->last = last;
    Py_DECREF(tmp); // must *NOT* decrement reference counter before assign
  }
  return 0;
}

static PyMemberDef Noddy_members[] = {
  {"number", T_INT, offsetof(Noddy, number), 0, "noddy number"},
  {NULL} // Sentinel
};

static PyObject *Noddy_getfirst(Noddy *self, void *closure)
{
  Py_INCREF(self->first);
  return self->first;
}

static int Noddy_setfirst(Noddy *self, PyObject *value, void *closure)
{
  if(!value){
    PyErr_SetString(PyExc_TypeError, "Cannot delete the first attribute");
    return -1;
  }
  if(!PyString_Check(value)){
    PyErr_SetString(PyExc_TypeError, "The first attribute must be a string");
    return -1;
  }
  Py_DECREF(self->first);
  Py_INCREF(value);
  self->first = value;
  return 0;
}

static PyObject *Noddy_getlast(Noddy *self, void *closure)
{
  Py_INCREF(self->last);
  return self->last;
}

static int Noddy_setlast(Noddy *self, PyObject *value, void *closure)
{
  if(!value){
    PyErr_SetString(PyExc_TypeError, "Cannot delete the last attribute");
    return -1;
  }
  if(!PyString_Check(value)){
    PyErr_SetString(PyExc_TypeError, "The last attribute must be a string");
    return -1;
  }
  Py_DECREF(self->last);
  Py_INCREF(value);
  self->last = value;
  return 0;
}

static PyGetSetDef Noddy_getseters[] = {
  {"first", (getter)Noddy_getfirst, (setter)Noddy_setfirst, "firstname", NULL},
  {"last", (getter)Noddy_getlast, (setter)Noddy_setlast, "lastname", NULL},
  {NULL} // Sentinel
};

static PyObject *Noddy_name(Noddy *self)
{
  static PyObject *format = NULL;
  PyObject *args, *result;
  if(!format){
    format = PyString_FromString("%s %s");
    if(!format) return NULL;
  }
#if 0
  if(!self->first){
    PyErr_SetString(PyExc_AttributeError, "first");
    return NULL;
  }
  if(!self->last){
    PyErr_SetString(PyExc_AttributeError, "last");
    return NULL;
  }
#endif
  args = Py_BuildValue("OO", self->first, self->last);
  if(!args) return NULL;
  result = PyString_Format(format, args);
  Py_DECREF(args);
  return result;
}

static PyMethodDef Noddy_methods[] = {
  {"name", (PyCFunction)Noddy_name,
    METH_NOARGS, "Return the name, combining the first and last name"},
  {NULL} // Sentinel
};

static PyTypeObject NoddyType = {
  PyObject_HEAD_INIT(NULL)  // PyObject_HEAD_INIT(&PyType_Type) <- PyType_Ready
  0,                        // ob_size
  "cpymatplotlib.Noddy",    // tp_name
  sizeof(Noddy),            // tp_basicsize
  0,                        // tp_itemsize
  (destructor)Noddy_dealloc, // tp_dealloc
  0,                        // tp_print
  0,                        // tp_getattr
  0,                        // tp_setattr
  0,                        // tp_compare
  0,                        // tp_repr
  0,                        // tp_as_number
  0,                        // tp_as_sequence
  0,                        // tp_as_mapping
  0,                        // tp_hash
  0,                        // tp_call
  0,                        // tp_str
  0,                        // tp_getattro
  0,                        // tp_setattro
  0,                        // tp_as_buffer
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, // tp_flags
  "Noddy objects",          // tp_doc
  (traverseproc)Noddy_traverse, // tp_traverse
  (inquiry)Noddy_clear,     // tp_clear
  0,                        // tp_richcompare
  0,                        // tp_weaklistoffset
  0,                        // tp_iter
  0,                        // tp_iternext
  Noddy_methods,            // tp_methods
  Noddy_members,            // tp_members
  Noddy_getseters,          // tp_getset
  0,                        // tp_base
  0,                        // tp_dict
  0,                        // tp_descr_get
  0,                        // tp_descr_set
  0,                        // tp_dictoffset
  (initproc)Noddy_init,     // tp_init
  0,                        // tp_alloc
  Noddy_new,                // tp_new
// and other fields
};

typedef struct _Nobject {
  PyObject_HEAD
  PyObject *a;
  PyObject *b;
  PyObject *c;
} Nobject;

static void Nobject_dealloc(Nobject *self)
{
  Py_XDECREF(self->a);
  Py_XDECREF(self->b);
  Py_XDECREF(self->c);
  self->ob_type->tp_free((PyObject *)self);
}

static PyObject *Nobject_new(PyTypeObject *typ, PyObject *args, PyObject *kw)
{
  // may call typ->tp_base->tp_new(...);
  // or PyObject_Call(PyObject_GetAttrString(super, "__new__"), args, kw);
  Nobject *self = (Nobject *)typ->tp_alloc(typ, 0);
  if(!self) return NULL;
  self->a = PyString_FromString("");
  if(!self->a){ Py_DECREF(self); return NULL; }
  self->b = PyString_FromString("");
  if(!self->b){ Py_DECREF(self); return NULL; }
  self->c = PyString_FromString("");
  if(!self->c){ Py_DECREF(self); return NULL; }
  return (PyObject *)self;
}

static int Nobject_init(Nobject *self, PyObject *args, PyObject *kw)
{
  // may call typ->tp_base->tp_init(...);
  // or PyObject_Call(PyObject_GetAttrString(super, "__init__"), args, kw);
  // but with self ?
  static char *ks[] = {"a", "b", "c", NULL};
  PyObject *a = NULL, *b = NULL, *c = NULL, *tmp;
  if(!PyArg_ParseTupleAndKeywords(args, kw, "|OOO", ks, &a, &b, &c))
    return -1;
  if(a){
    tmp = self->a;
    Py_INCREF(a);
    self->a = a;
    Py_XDECREF(tmp); // must *NOT* decrement reference counter before assign
  }
  if(b){
    tmp = self->b;
    Py_INCREF(b);
    self->b = b;
    Py_XDECREF(tmp); // must *NOT* decrement reference counter before assign
  }
  if(c){
    tmp = self->c;
    Py_INCREF(c);
    self->c = c;
    Py_XDECREF(tmp); // must *NOT* decrement reference counter before assign
  }
  return 0;
}

static PyMemberDef Nobject_members[] = {
  {"a", T_OBJECT_EX, offsetof(Nobject, a), 0, "builtin a"},
  {"b", T_OBJECT_EX, offsetof(Nobject, b), 0, "builtin b"},
  {"c", T_OBJECT_EX, offsetof(Nobject, c), 0, "builtin c"},
  {NULL} // Sentinel
};

static PyObject *Nobject_plain(Nobject *self)
{
  static PyObject *format = NULL;
  PyObject *args, *result;
  if(!format){
    format = PyString_FromString("%s %s %s");
    if(!format) return NULL;
  }
  if(!self->a){
    PyErr_SetString(PyExc_AttributeError, "a");
    return NULL;
  }
  if(!self->b){
    PyErr_SetString(PyExc_AttributeError, "b");
    return NULL;
  }
  if(!self->c){
    PyErr_SetString(PyExc_AttributeError, "c");
    return NULL;
  }
  args = Py_BuildValue("OOO", self->a, self->b, self->c);
  if(!args) return NULL;
  result = PyString_Format(format, args);
  Py_DECREF(args);
  return result;
}

static PyMethodDef Nobject_methods[] = {
  {"plain", (PyCFunction)Nobject_plain,
    METH_NOARGS, "Return the plain text, combining attributes"},
  {NULL} // Sentinel
};

static PyTypeObject NobjectType = {
  PyObject_HEAD_INIT(NULL)  // PyObject_HEAD_INIT(&PyType_Type) <- PyType_Ready
  0,                        // ob_size
  "cpymatplotlib.Nobject",  // tp_name
  sizeof(Nobject),          // tp_basicsize
  0,                        // tp_itemsize
  (destructor)Nobject_dealloc, // tp_dealloc
  0,                        // tp_print
  0,                        // tp_getattr
  0,                        // tp_setattr
  0,                        // tp_compare
  0,                        // tp_repr
  0,                        // tp_as_number
  0,                        // tp_as_sequence
  0,                        // tp_as_mapping
  0,                        // tp_hash
  0,                        // tp_call
  0,                        // tp_str
  0,                        // tp_getattro
  0,                        // tp_setattro
  0,                        // tp_as_buffer
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags
  "Nobject objects",        // tp_doc
  0,                        // tp_traverse
  0,                        // tp_clear
  0,                        // tp_richcompare
  0,                        // tp_weaklistoffset
  0,                        // tp_iter
  0,                        // tp_iternext
  Nobject_methods,          // tp_methods
  Nobject_members,          // tp_members
  0,                        // tp_getset
  0,                        // tp_base
  0,                        // tp_dict
  0,                        // tp_descr_get
  0,                        // tp_descr_set
  0,                        // tp_dictoffset
  (initproc)Nobject_init,   // tp_init
  0,                        // tp_alloc
  Nobject_new,              // tp_new
// and other fields
};

typedef struct _Abject {
//PyObject_HEAD             // built-in 'object' but nothing to inherit
//PyType_Type typ;          // built-in 'type'
//PyBaseObject_Type obj;    // built-in 'object'
//PySuper_Type sup;         // built-in 'super'
  Nobject nobj;             // inherits Nobject
} Abject;

static PyTypeObject AbjectType = {
  PyObject_HEAD_INIT(NULL)  // PyObject_HEAD_INIT(&PyType_Type) <- PyType_Ready
  0,                        // ob_size
  "cpymatplotlib.Abject",   // tp_name
  sizeof(Abject),           // tp_basicsize
  0,                        // tp_itemsize
  0,                        // tp_dealloc
  0,                        // tp_print
  0,                        // tp_getattr
  0,                        // tp_setattr
  0,                        // tp_compare
  0,                        // tp_repr
  0,                        // tp_as_number
  0,                        // tp_as_sequence
  0,                        // tp_as_mapping
  0,                        // tp_hash
  0,                        // tp_call
  0,                        // tp_str
  0,                        // tp_getattro
  0,                        // tp_setattro
  0,                        // tp_as_buffer
  Py_TPFLAGS_DEFAULT,       // tp_flags
  "Abject objects",         // tp_doc
// and other fields
};

static PyMethodDef cpymatplotlib_methods[] = {
  {"cpymPyObject", (PyCFunction)cpymPyObject,
    METH_VARARGS | METH_KEYWORDS, "*args, **kw:\n"
    " i:\n"
    " d:\n"
    " s:\n"
    " a:\n"
    "result: dict"},
  {"cpymFunc", (PyCFunction)cpymFunc,
    METH_VARARGS, "*args:\n"
    " i:\n"
    " d:\n"
    " s:\n"
    "result: dict"},
  {"cpymFuncKwArgs", (PyCFunction)cpymFuncKwArgs,
    METH_VARARGS | METH_KEYWORDS, "*args, **kw:\n"
    " i:\n"
    " d:\n"
    " s:\n"
    " a:\n"
    "result: dict"},
  {"cpymFuncNoArgs", (PyCFunction)cpymFuncNoArgs,
    METH_NOARGS, "no args:\n"
    "result: always fixed msg."},
  {"npCos", (PyCFunction)npCos, METH_VARARGS,
    "evaluate the cosine on a numpy array"},
  {"npLissajous", (PyCFunction)npLissajous, METH_VARARGS,
    "X-Y lissajous on a numpy array"},
  {NULL, NULL, 0, NULL}
};

static char cpymatplotlib_docstr[] = \
  "about this module\ntestdocstr\ntestdocstr\ntestdocstr";

PyMODINIT_FUNC initcpymatplotlib()
{
  // NoddyType.tp_new = PyType_GenericNew;
  if(PyType_Ready(&NoddyType) < 0) return;
  // NobjectType.tp_new = PyType_GenericNew;
  if(PyType_Ready(&NobjectType) < 0) return;
  AbjectType.tp_new = PyType_GenericNew;
  if(PyType_Ready(&AbjectType) < 0) return;
  PyObject *m = Py_InitModule3("cpymatplotlib",
    cpymatplotlib_methods, cpymatplotlib_docstr);
  if(!m) return;
  /* IMPORTANT: this must be called */
  import_array();
  Py_INCREF(&NoddyType);
  PyModule_AddObject(m, "Noddy", (PyObject *)&NoddyType);
  Py_INCREF(&NobjectType);
  PyModule_AddObject(m, "Nobject", (PyObject *)&NobjectType);
  Py_INCREF(&AbjectType);
  PyModule_AddObject(m, "Abject", (PyObject *)&AbjectType);
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
