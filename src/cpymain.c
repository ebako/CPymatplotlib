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
        if(!PyArg_Parse(o, "s", &s)){
          fprintf(stderr, "cannot parse StringObject\n");
        }else{
          fprintf(stdout, "resultPO['s']: [%s]\n", s);
        }
      }
      Py_DECREF(po);
    }
  }
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
