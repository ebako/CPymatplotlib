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

#define BUFSIZE 4096

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
    PyObject *po = PyObject_CallMethod(cpymatplotlib,
      "testPyObject", "ids", 511, 255.0, "abc");
    if(!po){
      fprintf(stderr, "cannot call testPyObject\n");
    }else{
      PyObject *o = PyDict_GetItemString(po, "s");
      if(!o){
        fprintf(stderr, "cannot get po['s']\n");
      }else{
        char *s;
        Py_INCREF(o);
        if(!PyArg_Parse(o, "s", &s)){ // s = PyString_AsString(o);
          fprintf(stderr, "cannot parse StringObject\n");
        }else{
          fprintf(stdout, "resultPO['s']: [%s]\n", s);
        }
      }
      Py_DECREF(po);
    }
  }
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
    PyObject *kw = Py_BuildValue("{sisi}",
      "debug", 0, "server_max_key_length", 999);
    mem = PyObject_CallMethod(memcache, "Client", "OO", lst, kw);
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
