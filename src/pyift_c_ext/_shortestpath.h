
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

PyObject *_dynamicArcWeightGridRoot(PyArrayObject *image, PyArrayObject *seeds);
PyObject *_dynamicArcWeightGridLabel(PyArrayObject *image, PyArrayObject *seeds);
PyObject *_dynamicArcWeightGridExpDecay(PyArrayObject *image, PyArrayObject *seeds, double alpha);

PyObject *_euclideanDistanceTransformGrid(PyArrayObject *_mask, PyArrayObject *_scales);

PyObject *_orientedSeedCompetitionGrid(PyArrayObject *_image, PyArrayObject *_seeds, PyArrayObject *_mask,
                                       double alpha, int background_label, double handicap);

PyObject *_watershedFromMinimaGrid(PyArrayObject *_image, PyArrayObject *_mask, PyArrayObject *_H_minima,
                                   double compactness, PyArrayObject *_scales);

#ifdef __cplusplus
}
#endif

#endif // _SHORTESTPATH_H
