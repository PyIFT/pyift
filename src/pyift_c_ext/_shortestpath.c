#include "_shortestpath.h"
#include "core.h"
#include "adjacency.h"
#include "heap.h"

#include <float.h>


static inline double square(double x) {
    return x * x;
}


PyObject *_seedCompetitionGrid(PyArrayObject *_image, PyArrayObject *_seeds)
{
    if (PyArray_NDIM(_image) != PyArray_NDIM(_seeds) + 1) {
        PyErr_SetString(PyExc_TypeError, "`seeds` ndarray must be one dimension lower than `image`.");
        return NULL;
    }

    PyArrayObject *image = NULL, *seeds = NULL;
    image = (PyArrayObject*) PyArray_FROM_OTF((PyObject*) _image, NPY_DOUBLE, NPY_ARRAY_CARRAY_RO);
    seeds = (PyArrayObject*) PyArray_FROM_OTF((PyObject*) _seeds, NPY_LONG, NPY_ARRAY_CARRAY_RO);
    if (!image || !seeds) goto err1;

    Coord shape;
    Adjacency *adj = NULL;
    int d = -1;
    if (PyArray_NDIM(image) == 3) {
        shape.z = 1;
        shape.y = PyArray_DIM(image, 0);
        shape.x = PyArray_DIM(image, 1);
        d = PyArray_DIM(image, 2);
        adj = circularAdjacency(1.0);
    } else if (PyArray_NDIM(image) == 4) {
        shape.z = PyArray_DIM(image, 0);
        shape.y = PyArray_DIM(image, 1);
        shape.x = PyArray_DIM(image, 2);
        d = PyArray_DIM(image, 3);
        adj = sphericAdjacency(1.0);
    } else {
        PyErr_SetString(PyExc_TypeError, "`seedCompetitionGrid` expected input with 3 or 4 number of dimensions.");
        goto err2;
    }

    if (!adj) goto err2;

    int size = shape.x * shape.y * shape.z;

    PyArrayObject *costs = NULL, *roots = NULL, *preds = NULL, *labels = NULL;

    costs = (PyArrayObject*) PyArray_SimpleNew(PyArray_NDIM(seeds), PyArray_DIMS(seeds), NPY_DOUBLE);
    roots = (PyArrayObject*) PyArray_SimpleNew(PyArray_NDIM(seeds), PyArray_DIMS(seeds), NPY_LONG);
    preds = (PyArrayObject*) PyArray_SimpleNew(PyArray_NDIM(seeds), PyArray_DIMS(seeds), NPY_LONG);
    labels = (PyArrayObject*) PyArray_SimpleNew(PyArray_NDIM(seeds), PyArray_DIMS(seeds), NPY_LONG);
    if (!costs || !roots || !preds || !labels) goto err3;

    double *image_ptr = PyArray_DATA(image);
    long *seeds_ptr = PyArray_DATA(seeds);
    double *costs_ptr = PyArray_DATA(costs);
    long *roots_ptr = PyArray_DATA(roots);
    long *preds_ptr = PyArray_DATA(preds);
    long *labels_ptr = PyArray_DATA(labels);
    
    Heap *heap = createHeap(size, costs_ptr);
    if (!heap) goto err3;

    for (int i = 0; i < size; i++)
    {
        roots_ptr[i] = i;
        preds_ptr[i] = -1;
        if (seeds_ptr[i] > 0) {
            costs_ptr[i] = 0;
            labels_ptr[i] = seeds_ptr[i];
            insertHeap(heap, i);
        } else {
            costs_ptr[i] = DBL_MAX;
            labels_ptr[i] = -1;
        }
    }

    while (!emptyHeap(heap))
    {
        int p = popHeap(heap);
        int p_stride = p * d;
        Coord u = indexToCoord(&shape, p);
        for (int i = 1; i < adj->size; i++)
        {
            Coord v = adjacentCoord(&u, adj, i);
            if (validCoord(&shape, &v))
            {
                int q = coordToIndex(&shape, &v);
                if (heap->colors[q] != BLACK)
                {
                    int q_stride = q * d;
                    double dist = 0;
                    for (int j = 0; j < d; j++)
                        dist += square(image_ptr[p_stride + j] - image_ptr[q_stride + j]);
                    dist = sqrt(dist);
                    dist = PyArray_MAX(dist, costs_ptr[p]);
                    if (dist < costs_ptr[q])
                    {
                        roots_ptr[q] = roots_ptr[p];
                        labels_ptr[q] = labels_ptr[p];
                        preds_ptr[q] = p;
                        costs_ptr[q] = dist;
                        if (heap->colors[q] == WHITE)
                            insertHeap(heap, q);
                        else
                            goUpHeap(heap, q);
                    }
                }
            }
        }
    }

    destroyHeap(&heap);
    destroyAdjacency(&adj);
    Py_DECREF(seeds);
    Py_DECREF(image);

    return Py_BuildValue("(NNNN)", costs, roots, preds, labels);

    // error handling
    err3:
    Py_XDECREF(labels);
    Py_XDECREF(preds);
    Py_XDECREF(roots);
    Py_XDECREF(costs);
    err2:
    destroyAdjacency(&adj);
    PyErr_NoMemory();
    err1:
    Py_XDECREF(seeds);
    Py_XDECREF(image);
    return NULL;
}


PyObject *_seedCompetitionGraph(PyArrayObject *_weights, PyArrayObject *_indices, PyArrayObject *_indptr, PyArrayObject *_seeds)
{
    if (PyArray_NDIM(_weights) != 1 || PyArray_NDIM(_indices) != 1 || PyArray_NDIM(_indptr) != 1 || PyArray_NDIM(_seeds) != 1) {
        PyErr_SetString(PyExc_TypeError, "`weights`, `indices`, `indptr` and `seeds` ndarrays must be one dimensional.");
        return NULL;
    }

    if (PyArray_DIM(_seeds, 0) != (PyArray_DIM(_indptr, 0) - 1)) {
        PyErr_SetString(PyExc_TypeError, "`indptr` length must be one less than `seeds` length.");
        return NULL;
    }

    if (PyArray_DIM(_weights, 0) != PyArray_DIM(_indices, 0)) {
        PyErr_SetString(PyExc_TypeError, "`weights` and `indices` must have the same length.");
        return NULL;
    }

    PyArrayObject *weights = NULL, *indices = NULL, *indptr = NULL, *seeds = NULL;
    weights = (PyArrayObject*) PyArray_FROM_OTF((PyObject*) _weights, NPY_DOUBLE, NPY_ARRAY_CARRAY_RO);
    indices = (PyArrayObject*) PyArray_FROM_OTF((PyObject*) _indices, NPY_LONG, NPY_ARRAY_CARRAY_RO);
    indptr = (PyArrayObject*) PyArray_FROM_OTF((PyObject*) _indptr, NPY_LONG, NPY_ARRAY_CARRAY_RO);
    seeds = (PyArrayObject*) PyArray_FROM_OTF((PyObject*) _seeds, NPY_LONG, NPY_ARRAY_CARRAY_RO);
    if (!weights || !indices || !indptr || !seeds) goto err1;

    PyArrayObject *costs = NULL, *roots = NULL, *preds = NULL, *labels = NULL;
    roots = (PyArrayObject*) PyArray_SimpleNew(1, PyArray_DIMS(seeds), NPY_LONG);
    costs = (PyArrayObject*) PyArray_SimpleNew(1, PyArray_DIMS(seeds), NPY_DOUBLE);
    preds = (PyArrayObject*) PyArray_SimpleNew(1, PyArray_DIMS(seeds), NPY_LONG);
    labels = (PyArrayObject*) PyArray_SimpleNew(1, PyArray_DIMS(seeds), NPY_LONG);
    if (!roots || !costs || !preds || !labels) goto err2;

    double *weights_ptr = PyArray_DATA(weights);
    long *indices_ptr = PyArray_DATA(indices);
    long *indptr_ptr = PyArray_DATA(indptr);
    long *seeds_ptr = PyArray_DATA(seeds);
    double *costs_ptr = PyArray_DATA(costs);
    long *roots_ptr = PyArray_DATA(roots);
    long *preds_ptr = PyArray_DATA(preds);
    long *labels_ptr = PyArray_DATA(labels);
    
    long size = PyArray_DIM(seeds, 0);
    Heap *heap = createHeap(size, costs_ptr);
    if (!heap) goto err2;

    for (int i = 0; i < size; i++)
    {
        roots_ptr[i] = i;
        preds_ptr[i] = -1;
        if (seeds_ptr[i] > 0) {
            costs_ptr[i] = 0;
            labels_ptr[i] = seeds_ptr[i];
            insertHeap(heap, i);
        } else {
            costs_ptr[i] = DBL_MAX;
            labels_ptr[i] = -1;
        }
    }

    while (!emptyHeap(heap))
    {
        int p = popHeap(heap);
        for (int i = indptr_ptr[p]; i < indptr_ptr[p + 1]; i++)
        {
            int q = indices_ptr[i];
            if (heap->colors[q] != BLACK)
            {
                double dist = PyArray_MAX(weights_ptr[i], costs_ptr[p]);
                if (dist < costs_ptr[q])
                {
                    roots_ptr[q] = roots_ptr[p];
                    labels_ptr[q] = labels_ptr[p];
                    preds_ptr[q] = p;
                    costs_ptr[q] = dist;
                    if (heap->colors[q] == WHITE)
                        insertHeap(heap, q);
                    else
                        goUpHeap(heap, q);
                }
            }
        }
    }

    destroyHeap(&heap);
    Py_DECREF(seeds);
    Py_DECREF(indptr);
    Py_DECREF(indices);
    Py_DECREF(weights);

    return Py_BuildValue("(NNNN)", costs, roots, preds, labels);

    // error handling
    err2:
    PyErr_NoMemory();
    Py_XDECREF(labels);
    Py_XDECREF(preds);
    Py_XDECREF(roots);
    Py_XDECREF(costs);
    err1:
    Py_XDECREF(seeds);
    Py_XDECREF(indptr);
    Py_XDECREF(indices);
    Py_XDECREF(weights);
    return NULL;
}


static void free_double_matrix(double **matrix, int length)
{
    if (!matrix) return;
    for (int i = 0; i < length; i++) {
        free(matrix[i]);
        matrix[i] = NULL;
    }
    free(matrix);
}


PyObject *_dynamicArcWeightGrid(PyArrayObject *_image, PyArrayObject *_seeds)
{
    if (PyArray_NDIM(_image) != PyArray_NDIM(_seeds) + 1) {
        PyErr_SetString(PyExc_TypeError, "`seeds` ndarray must be one dimension lower than `image`.");
        return NULL;
    }

    PyArrayObject *image = NULL, *seeds = NULL;
    image = (PyArrayObject*) PyArray_FROM_OTF((PyObject*) _image, NPY_DOUBLE, NPY_ARRAY_CARRAY_RO);
    seeds = (PyArrayObject*) PyArray_FROM_OTF((PyObject*) _seeds, NPY_LONG, NPY_ARRAY_CARRAY_RO);
    if (!image || !seeds) goto err1;

    Coord shape;
    Adjacency *adj = NULL;
    int d = -1;
    if (PyArray_NDIM(image) == 3) {
        shape.z = 1;
        shape.y = PyArray_DIM(image, 0);
        shape.x = PyArray_DIM(image, 1);
        d = PyArray_DIM(image, 2);
        adj = circularAdjacency(1.0);
    } else if (PyArray_NDIM(image) == 4) {
        shape.z = PyArray_DIM(image, 0);
        shape.y = PyArray_DIM(image, 1);
        shape.x = PyArray_DIM(image, 2);
        d = PyArray_DIM(image, 3);
        adj = sphericAdjacency(1.0);
    } else {
        PyErr_SetString(PyExc_TypeError, "`dynamicArcWeightGrid` expected input with 3 or 4 number of dimensions.");
        goto err1;
    }
    if (!adj) goto err2;

    int size = shape.x * shape.y * shape.z;

    int *tree_sizes = NULL;
    double **tree_avgs = NULL;
    PyArrayObject *costs = NULL, *roots = NULL, *preds = NULL, *labels = NULL;

    tree_sizes = calloc(size, sizeof *tree_sizes);
    tree_avgs = calloc(size, sizeof *tree_avgs);
    if (!tree_sizes || !tree_avgs) goto err3;

    costs = (PyArrayObject*) PyArray_SimpleNew(PyArray_NDIM(seeds), PyArray_DIMS(seeds), NPY_DOUBLE);
    roots = (PyArrayObject*) PyArray_SimpleNew(PyArray_NDIM(seeds), PyArray_DIMS(seeds), NPY_LONG);
    preds = (PyArrayObject*) PyArray_SimpleNew(PyArray_NDIM(seeds), PyArray_DIMS(seeds), NPY_LONG);
    labels = (PyArrayObject*) PyArray_SimpleNew(PyArray_NDIM(seeds), PyArray_DIMS(seeds), NPY_LONG);
    if (!costs || !roots || !preds || !labels) goto err3;

    double *image_ptr = PyArray_DATA(image);
    long *seeds_ptr = PyArray_DATA(seeds);
    double *costs_ptr = PyArray_DATA(costs);
    long *roots_ptr = PyArray_DATA(roots);
    long *preds_ptr = PyArray_DATA(preds);
    long *labels_ptr = PyArray_DATA(labels);

    Heap *heap = createHeap(size, costs_ptr);
    if (!heap) goto err3;

    for (int i = 0; i < size; i++)
    {
        roots_ptr[i] = i;
        preds_ptr[i] = -1;
        if (seeds_ptr[i] > 0) {
            costs_ptr[i] = 0;
            labels_ptr[i] = seeds_ptr[i];
            tree_avgs[i] = calloc(d, sizeof **tree_avgs);
            if (!tree_avgs[i]) goto err4;
            insertHeap(heap, i);
        } else {
            costs_ptr[i] = DBL_MAX;
            labels_ptr[i] = -1;
        }
    }

    while (!emptyHeap(heap))
    {
        int p = popHeap(heap);
        int p_stride = p * d;

        int r = roots_ptr[p];
        double *tree_avg = tree_avgs[r];
        tree_sizes[r] += 1;
        for (int j = 0; j < d; j++) {
            tree_avg[j] += (image_ptr[p_stride + j] - tree_avg[j]) / tree_sizes[r];
        }

        Coord u = indexToCoord(&shape, p);
        for (int i = 1; i < adj->size; i++)
        {
            Coord v = adjacentCoord(&u, adj, i);
            if (validCoord(&shape, &v))
            {
                int q = coordToIndex(&shape, &v);
                if (heap->colors[q] != BLACK)
                {
                    int q_stride = q * d;
                    double dist = 0;
                    for (int j = 0; j < d; j++)
                        dist += square(tree_avg[j] - image_ptr[q_stride + j]);
                    dist = sqrt(dist);
                    dist = PyArray_MAX(dist, costs_ptr[p]);
                    if (dist < costs_ptr[q])
                    {
                        roots_ptr[q] = r;
                        labels_ptr[q] = labels_ptr[p];
                        preds_ptr[q] = p;
                        costs_ptr[q] = dist;
                        if (heap->colors[q] == WHITE)
                            insertHeap(heap, q);
                        else
                            goUpHeap(heap, q);
                    }
                }
            }
        }
    }

    PyObject *avgs = PyDict_New();
    if (!avgs) goto err4;
    npy_intp tree_d[1] = {d};
    for (int i = 0; i < size; i++)
    {
        if (tree_avgs[i])
        {
            PyObject *avg = PyArray_SimpleNewFromData(1, tree_d, NPY_DOUBLE, tree_avgs[i]);
            PyArray_ENABLEFLAGS((PyArrayObject*) avg, NPY_ARRAY_OWNDATA);
            Coord _coord = indexToCoord(&shape, i);
            PyObject *coord = NULL;
            if (shape.z == 1)
                coord = Py_BuildValue("(NN)", PyLong_FromLong(_coord.y), PyLong_FromLong(_coord.x));
            else
                coord = Py_BuildValue("(NNN)", PyLong_FromLong(_coord.z),
                                      PyLong_FromLong(_coord.y), PyLong_FromLong(_coord.x));
            PyObject *pair = Py_BuildValue("(NN)", PyLong_FromLong(tree_sizes[i]), avg);
            PyDict_SetItem(avgs, coord, pair);
        }
    }

    free(tree_avgs);
    free(tree_sizes);
    destroyHeap(&heap);
    destroyAdjacency(&adj);
    Py_DECREF(seeds);
    Py_DECREF(image);

    return Py_BuildValue("(NNNNN)", costs, roots, preds, labels, avgs);

    // error handling
    err4: destroyHeap(&heap);
    err3:
    Py_XDECREF(labels);
    Py_XDECREF(preds);
    Py_XDECREF(roots);
    Py_XDECREF(costs);
    free(tree_sizes);
    free_double_matrix(tree_avgs, size);
    err2:
    destroyAdjacency(&adj);
    PyErr_NoMemory();
    err1:
    Py_XDECREF(seeds);
    Py_XDECREF(image);
    return NULL;
}
