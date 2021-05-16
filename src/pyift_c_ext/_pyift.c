#define PY_SSIZE_T_CLEAN
#include "Python.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PYIFT_ARRAY_API
#include "numpy/arrayobject.h"

#include "_livewire.h"
#include "_shortestpath.h"


PyObject *euclidean_distance_transform_grid(PyObject *self, PyObject *args)
{
    PyArrayObject *mask = NULL, *scales = NULL;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &mask, &PyArray_Type, &scales))
        return NULL;
    return _euclideanDistanceTransformGrid(mask, scales);
}


PyObject *dynamic_arc_weight_grid_exp_decay(PyObject *self, PyObject *args)
{
    PyArrayObject *image = NULL, *seeds = NULL;
    double alpha = 0;
    if (!PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &image, &PyArray_Type, &seeds, &alpha))
        return NULL;
    return _dynamicArcWeightGridExpDecay(image, seeds, alpha);
}


PyObject *dynamic_arc_weight_grid_label(PyObject *self, PyObject *args)
{
    PyArrayObject *image = NULL, *seeds = NULL;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &image, &PyArray_Type, &seeds))
        return NULL;
    return _dynamicArcWeightGridLabel(image, seeds);
}


PyObject *dynamic_arc_weight_grid_root(PyObject *self, PyObject *args)
{
    PyArrayObject *image = NULL, *seeds = NULL;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &image, &PyArray_Type, &seeds))
        return NULL;
    return _dynamicArcWeightGridRoot(image, seeds);
}


PyObject *livewire_path(PyObject *self, PyObject *args)
{
    PyArrayObject *image = NULL, *costs = NULL, *preds = NULL, *labels = NULL, *saliency = NULL;
    char *livewire_fun_str = NULL;
    int src = -1, dst = -1;
    double param1 = 1, param2 = 1;
    if (PyTuple_Size(args) == 8) {
        if (!PyArg_ParseTuple(args, "O!O!O!O!sdii", &PyArray_Type, &image, &PyArray_Type, &costs,
                              &PyArray_Type, &preds, &PyArray_Type, &labels, &livewire_fun_str,
                              &param1, &src, &dst))
            return NULL;
    } else if (PyTuple_Size(args) == 10) {
        if (!PyArg_ParseTuple(args, "O!O!O!O!O!sddii", &PyArray_Type, &image, &PyArray_Type, &saliency,
                              &PyArray_Type, &costs, &PyArray_Type, &preds, &PyArray_Type,
                              &labels, &livewire_fun_str, &param1, &param2, &src, &dst))
            return NULL;
    } else {
         PyErr_SetString(PyExc_AttributeError, "Invalid arguments for live-wire C api.");
         return NULL;
    }

    liveWireFun livewire_fun = -1;
    if (!strcmp(livewire_fun_str, "exp"))
        livewire_fun = EXP;
    else if (!strcmp(livewire_fun_str, "exp-saliency")) {
        livewire_fun = EXPSAL;
        if (!saliency) {
            PyErr_SetString(PyExc_AttributeError, "`saliency` must be supplied when using `exp-saliency` arc-weight.");
            return NULL;
        }
    }

    return _livewirePath(image, saliency, costs, preds, labels, livewire_fun, param1, param2, src, dst);
}


PyObject *seed_competition_graph(PyObject *self, PyObject *args)
{
    PyArrayObject *weights = NULL, *indices = NULL, *indptr = NULL, *seeds = NULL;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &weights, &PyArray_Type, &indices,
                          &PyArray_Type, &indptr, &PyArray_Type, &seeds))
        return NULL;
    return _seedCompetitionGraph(weights, indices, indptr, seeds);
}



PyObject *seed_competition_grid(PyObject *self, PyObject *args)
{
    PyArrayObject *image = NULL, *seeds = NULL;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &image, &PyArray_Type, &seeds))
        return NULL;
    return _seedCompetitionGrid(image, seeds);
}


PyObject *watershed_from_minima_grid(PyObject *self, PyObject *args)
{
    PyArrayObject *image = NULL, *mask = NULL;
    double penalization = 1;
    if (!PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &image, &PyArray_Type, &mask, &penalization))
        return NULL;
    return _watershedFromMinimaGrid(image, mask, penalization);
}


// alphabetical order
static PyMethodDef functions[] = {
    {"euclidean_distance_transform_grid", (PyCFunction) euclidean_distance_transform_grid, METH_VARARGS},
    {"dynamic_arc_weight_grid_exp_decay", (PyCFunction) dynamic_arc_weight_grid_exp_decay, METH_VARARGS},
    {"dynamic_arc_weight_grid_label", (PyCFunction) dynamic_arc_weight_grid_label, METH_VARARGS},
    {"dynamic_arc_weight_grid_root", (PyCFunction) dynamic_arc_weight_grid_root, METH_VARARGS},
    {"livewire_path", (PyCFunction) livewire_path, METH_VARARGS},
    {"seed_competition_graph", (PyCFunction) seed_competition_graph, METH_VARARGS},
    {"seed_competition_grid", (PyCFunction) seed_competition_grid, METH_VARARGS},
    {"watershed_from_minima_grid", (PyCFunction) watershed_from_minima_grid, METH_VARARGS},
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
