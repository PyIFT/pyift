#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PYIFT_ARRAY_API
#include "numpy/arrayobject.h"

#include "_shortestpath.h"


PyObject *seed_competition_grid(PyObject *self, PyObject *args)
{
    PyArrayObject *image = NULL, *seeds = NULL;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &image, &PyArray_Type, &seeds))
        return NULL;
    return _seedCompetitionGrid(image, seeds);
}


PyObject *seed_competition_graph(PyObject *self, PyObject *args)
{
    PyArrayObject *weights = NULL, *indices = NULL, *indptr = NULL, *seeds = NULL;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &weights, &PyArray_Type, &indices,
                          &PyArray_Type, &indptr, &PyArray_Type, &seeds))
        return NULL;
    return _seedCompetitionGraph(weights, indices, indptr, seeds);
}


PyObject *dynamic_arc_weight_grid(PyObject *self, PyObject *args)
{
    PyArrayObject *image = NULL, *seeds = NULL;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &image, &PyArray_Type, &seeds))
        return NULL;
    return _dynamicArcWeightGrid(image, seeds);
}


// alphabetical order
static PyMethodDef functions[] = {
    {"dynamic_arc_weight_grid", (PyCFunction) dynamic_arc_weight_grid, METH_VARARGS},
    {"seed_competition_grid", (PyCFunction) seed_competition_grid, METH_VARARGS},
    {"seed_competition_graph", (PyCFunction) seed_competition_graph, METH_VARARGS},
    {NULL, NULL} // sentinel
};


PyMODINIT_FUNC PyInit__pyift(void)
{
    import_array();

    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,  // m_base
        "_pyift",               // m_name
        NULL,                   // m_doc
        -1,                     // m_size
        functions,
    };

    PyObject *m = PyModule_Create(&module_def);

    return m;
}
