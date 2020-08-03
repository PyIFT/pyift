#ifndef _LIVEWIRE_H
#define _LIVEWIRE_H

#ifdef __cplusplus
extern "C" {
#endif

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PYIFT_ARRAY_API
#include "numpy/arrayobject.h"


typedef enum {
    EXP = 0,
    EXPSAL = 1,
} liveWireFun;


PyObject *_livewirePath(PyArrayObject *image, PyArrayObject *saliency, PyArrayObject *costs,
                        PyArrayObject *preds, PyArrayObject *labels, liveWireFun livewire_fun,
                        double param1, double param2, int src, int dst);


#ifdef __cplusplus
}
#endif

#endif // _LIVEWIRE_H
