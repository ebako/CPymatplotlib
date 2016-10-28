/*
  cpymain.c

  >mingw32-make -f makefile.tdmgcc64
*/

#ifndef UNICODE
#define UNICODE
#endif

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <numpy/arrayobject.h>
#include <math.h>

#define BUFSIZE 4096

void prepare_numpy()
{
  import_array(); // IMPORTANT: this must be called (return void)
}

int main(int ac, char **av)
{
  char buf[BUFSIZE];
  int i;
  if(sizeof(size_t) > 4){
    fprintf(stderr, "May be running 64bit version.\nskipped.\n");
    return 1;
  }
#if 0
  return Py_Main(ac, av);
#else
  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append('../dll')");
  PyRun_SimpleString("sys.stdout.write('%s\\n' % sys.version)");
  PyRun_SimpleString("sys.stdout.write('Hello, Python!\\n')");
  PyObject *cpymatplotlib = PyImport_ImportModule("cpymatplotlib");
  if(!cpymatplotlib){
    fprintf(stderr, "cannot import cpymatplotlib\n");
  }else{
    PyObject *tpl = Py_BuildValue("(ids)", 511, 255.0, "abc");
#if 1
#if 0
    PyObject *a = PyObject_CallObject(
      PyObject_GetAttrString(cpymatplotlib, "Abject"), NULL);
#else
    PyRun_SimpleString("class __A(object): pass");
    PyObject *m = PyImport_AddModule("__main__");
    PyObject *a = PyObject_CallObject(
      PyObject_GetAttrString(m, "__A"), NULL);
    if(a) Py_INCREF(a);
#endif
#else
#if 1
    PyObject *ini = PyTuple_New(0);
    PyObject *a = PyObject_Call(
      PyObject_GetAttrString(cpymatplotlib, "Nobject"), ini, NULL);
#else
    PyObject *a = PyObject_CallObject(
      PyObject_GetAttrString(cpymatplotlib, "Nobject"), NULL);
#endif
#endif
    if(!a){
      fprintf(stderr, "cannot call cpymatplotlib.Nobject\n");
    }else{
#if 0
      PyObject_SetAttrString(a, "a", PyInt_FromLong(123));
      PyObject_SetAttrString(a, "b", PyLong_FromLong(456));
      PyObject_SetAttrString(a, "c", PyString_FromString("enroute"));
#else
      PyObject *d = PyObject_GetAttrString(a, "__dict__");
      if(!d){
        fprintf(stderr, "cannot get a.__dict__\n");
      }else{
        PyDict_SetItemString(d, "a", PyInt_FromLong(123));
        PyDict_SetItemString(d, "b", PyLong_FromLong(456));
        PyDict_SetItemString(d, "c", PyString_FromString("enroute"));
      }
#endif
    }
    PyObject *kw = Py_BuildValue("{sO}", "a", a);
    PyObject *po = PyObject_Call(
      PyObject_GetAttrString(cpymatplotlib, "cpymPyObject"), tpl, kw);
    if(!po){
      fprintf(stderr, "cannot call cpymPyObject\n");
    }else{
      PyObject *o = PyDict_GetItemString(po, "o");
      if(!o){
        fprintf(stderr, "cannot get po['o']\n");
      }else{
        char *s = PyString_AsString(PyObject_CallMethod(o, "__str__", NULL));
        if(!s){
          fprintf(stderr, "cannot parse StringObject\n");
        }else{
          fprintf(stdout, "resultPO['o']: [%s]\n", s);
        }
      }
      Py_DECREF(po);
    }
  }
#if 0 // success
  PyRun_SimpleString("sys.path.append('../src')");
  PyRun_SimpleString("from test_cpymatplotlib import draw_realtime as dreal");
  PyRun_SimpleString("dreal(10)");
  // Fatal Python error: PyEval_RestoreThread: NULL tstate
#else
  PyObject *np = PyImport_ImportModule("numpy");
  prepare_numpy(); // IMPORTANT: this must be called (return void)
  PyObject *pylab = PyImport_ImportModule("pylab");
  if(!np || !pylab){
    fprintf(stderr, "cannot import numpy or pylab\n");
  }else{
    PyObject *fig = PyObject_CallMethod(pylab, "figure", NULL);
    PyObject *ax0 = PyObject_CallMethod(fig, "add_subplot", "i", 211);
    PyObject *ax1 = PyObject_CallMethod(fig, "add_subplot", "i", 212);
    if(!ax0 || !ax1){
      fprintf(stderr, "cannot add_subplot ax0 or ax1\n");
    }else{
      int ndim = 1;
      npy_intp dims[] = {10}; // [NPY_MAXDIMS]
#if 0
//    PyObject *x = PyObject_CallMethod(np, "arange", "ddd", 0.2, 0.4, 0.02);
      PyObject *x = PyArray_Arange(0.1, 0.5, 0.05, NPY_DOUBLE);
//    PyObject *x = PyArray_Arange(10, 50, 5, NPY_INT); // careful to (double)y
//    PyObject *x = PyArray_ZEROS(ndim, dims, NPY_DOUBLE, 0); // fortran:0
//    PyObject *x = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE); // no initialize
//    double d[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
//    dims[0] = sizeof(d) / sizeof(d[0]);
//    PyObject *x = PyArray_SimpleNewFromData(ndim, dims, NPY_DOUBLE, d);
      dims[0] = PyArray_DIMS(x)[0];
      PyObject *y = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE); // no initialize
#else
      PyObject *x = PyObject_CallMethod(np, "arange", "ddd", 0., 6.19, 0.02);
      dims[0] = PyArray_DIMS(x)[0];
      PyObject *y = PyObject_CallMethod(cpymatplotlib, "npLissajous",
        "Odd", x, 4., 3.);
#endif
      if(!x || !y){
        fprintf(stderr, "cannot allocate ndarray x or y\n");
      }else{
#if 0
        double *src = (double *)PyArray_DATA(x);
        double *dst = (double *)PyArray_DATA(y);
        int i;
        for(i = 0; i < PyArray_DIMS(x)[0]; ++i) *dst++ = exp(-10. * *src++);
#endif
        fprintf(stdout, "allocate ndarray [%d]\n", dims[0]);
        Py_INCREF(x);
        Py_INCREF(y);
        PyObject_CallMethod(ax0, "plot", "OO", x, y);
        PyObject_CallMethod(ax1, "plot", "OO", y, x);
      }
    }
    PyObject_CallMethod(pylab, "show", NULL);
  }
#endif
#if 0 // success
  char *lines[] = {
    "import memcache",
    "import httplib2",
    "mem = memcache.Client(['127.0.0.1:11211'],"\
     " debug=0, server_max_key_length=999)",
    "http = httplib2.Http(mem,"\
     " timeout=5, disable_ssl_certificate_validation=True)",
    "http.force_exception_to_status_code = True",
    "header, bdy = http.request('https://www.google.com/')",
    "sys.stdout.write('status: %s\\n' % header['status'])",
    "if header['status'] == '200': sys.stdout.write('body: %s\\n' % bdy)"};
  for(i = 0; i < sizeof(lines) / sizeof(lines[0]); ++i)
    PyRun_SimpleString(lines[i]);
#else // success (now solved the problem to call with **kwargs)
  PyObject *mem = NULL;
  PyObject *memcache = PyImport_ImportModule("memcache");
  if(!memcache){
    fprintf(stderr, "cannot import memcache\n");
  }else{
    PyObject *lst = Py_BuildValue("[s]", "127.0.0.1:11211");
    PyObject *tpl = PyTuple_Pack(1, lst); // Py_BuildValue("(O)", lst);
    PyObject *kw = Py_BuildValue("{sisi}",
      "debug", 0, "server_max_key_length", 999);
    mem = PyObject_Call(PyObject_GetAttrString(memcache, "Client"), tpl, kw);
    if(!mem) fprintf(stderr, "cannot get memcache.Client(...)\n");
  }
  PyObject *hb = NULL;
  PyObject *http = NULL;
  PyObject *httplib2 = PyImport_ImportModule("httplib2");
  if(!httplib2){
    fprintf(stderr, "cannot import httplib2\n");
  }else{
    PyObject *tpl = PyTuple_Pack(1, mem); // Py_BuildValue("(O)", mem);
    PyObject *kw = Py_BuildValue("{sisO}",
      "timeout", 5, "disable_ssl_certificate_validation", Py_True);
    http = PyObject_Call(PyObject_GetAttrString(httplib2, "Http"), tpl, kw);
//  http = PyObject_CallMethod(httplib2, "Http", "O", mem); // skip timeout/ssl
    if(!http) fprintf(stderr, "cannot get httplib2.Http(...)\n");
    // status:"408"=RequestTimeout instead of timeout exception
    PyObject *t = Py_True;
    Py_INCREF(t);
    PyObject_SetAttrString(http, "force_exception_to_status_code", t);
//  char *url = "http://localhost:8080/";
//  char *url = "http://www.google.com/";
    char *url = "https://www.google.com/";
    hb = PyObject_CallMethod(http, "request", "s", url);
    if(!hb) fprintf(stderr, "cannot get http.request(...)\n");
  }
  if(mem && http && hb){
    PyObject *header = PyTuple_GetItem(hb, 0);
    PyObject *status = PyDict_GetItemString(header, "status");
    fprintf(stdout, "status: %s\n", PyString_AsString(status)); // "500" "200"
    if(!strcmp(PyString_AsString(status), "200")){
      PyObject *bdy = PyTuple_GetItem(hb, 1);
      fprintf(stdout, "body: %s\n", PyString_AsString(bdy));
    }
  }
#endif
#if 0
  fprintf(stdout, ">>> ");
  while(fgets(buf, sizeof(buf), stdin)){
    PyRun_SimpleString(buf); // may be hidden stdout
    // PyRun_SimpleString("sys.stdout.write(str(v))");
    fprintf(stdout, ">>> ");
  }
#else
// import code
// ic = code.InteractiveConsole(locals())
// ic.push('a = 1')
// ic.push('a')
// ic.push('a + 2')
// ic.interact()
// # disabled exit() and quit() ?

  fprintf(stdout, "Python %s on %s\n", Py_GetVersion(), Py_GetPlatform());
  fprintf(stdout, "Type \"help\", \"copyright\", \"credits\" or \"license\" "\
    "for more information.\n");
  PyRun_SimpleString("import code");
  PyRun_SimpleString("__interpreter = code.InteractiveConsole(locals())");
  PyObject *mainModule = PyImport_AddModule("__main__");
  PyObject *globalDict = PyModule_GetDict(mainModule);
  PyObject *interpreter = PyDict_GetItemString(globalDict, "__interpreter");
  if(interpreter) Py_INCREF(interpreter);
  int multiline = 0;
  fprintf(stdout, ">>> ");
  while(fgets(buf, sizeof(buf), stdin)){
    int len = strlen(buf);
    if(!len) break;
    if(buf[len - 1] == 0x0A) buf[len - 1] = '\0';
    len = strlen(buf);
    if(buf[len - 1] == 0x0D) buf[len - 1] = '\0';
    // if(!(len = strlen(buf))) continue; // through because end of block
    if(!strcmp(buf, "exit()")) break;
    if(!strcmp(buf, "quit()")) break;
    PyObject *result = PyObject_CallMethod(interpreter, "push", "s", buf);
    if(!result){
      fprintf(stderr, "cannot call push('%s')\n", buf);
    }else{
      int status;
      if(PyArg_Parse(result, "i", &status)) multiline = status > 0;
      Py_DECREF(result);
    }
    fprintf(stdout, multiline ? "... " : ">>> ");
  }
#endif
  Py_Finalize();
  fprintf(stdout, "bye.\n");
  return 0;
#endif
}
