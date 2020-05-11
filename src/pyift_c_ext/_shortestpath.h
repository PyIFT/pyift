
#ifndef _SHORTESTPATH_H
#define _SHORTESTPATH_H

#ifdef __cplusplus
extern "C" {
#endif

#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PYIFT_ARRAY_API
#include "numpy/arrayobject.h"


PyObject *_seedCompetitionGrid(PyArrayObject *image, PyArrayObject *seeds);
PyObject *_seedCompetitionGraph(PyArrayObject *data, PyArrayObject *indices, PyArrayObject *indptr, PyArrayObject *seeds);

PyObject *_dynamicArcWeightGrid(PyArrayObject *image, PyArrayObject *seeds);


#ifdef __cplusplus
}
#endif

#endif // _SHORTESTPATH_H