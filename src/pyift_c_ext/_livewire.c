#include "_livewire.h"
#include "core.h"
#include "adjacency.h"
#include "heap.h"
#include "set.h"

#include <float.h>

#ifndef M_SQRT2
    #define M_SQRT2 1.41421356237309504880
#endif


static inline double square(double x) {
    return x * x;
}


static inline double distance(double *features, int p, int q, int d)
{
    int p_stride = p * d, q_stride = q * d;
    double dist = 0;
    for (int j = 0; j < d; j++)
        dist += square(features[p_stride + j] - features[q_stride + j]);
    dist = sqrt(dist);
    return dist;
}


PyObject *_livewirePath(PyArrayObject *image, PyArrayObject *saliency, PyArrayObject *costs,
                        PyArrayObject *preds, PyArrayObject *labels, liveWireFun livewire_fun,
                        double param1, double param2, int src, int dst)
{
    double *image_ptr = PyArray_DATA(image);
    double *costs_ptr = PyArray_DATA(costs);
    long *preds_ptr = PyArray_DATA(preds);
    bool *labels_ptr = PyArray_DATA(labels);

    double *saliency_ptr = NULL;
    int sal_d = 0;
    if (livewire_fun == EXPSAL)
    {
        if (!saliency)
            return NULL;
        saliency_ptr = PyArray_DATA(saliency);
        sal_d = PyArray_DIM(saliency, 2);
    }

    Coord shape = {.x = PyArray_DIM(image, 1),
                   .y = PyArray_DIM(image, 0),
                   .z = 1};

    int d = PyArray_DIM(image, 2), size = shape.x * shape.y * shape.z;
    Heap *heap = createHeap(size, costs_ptr);

    Adjacency *adj = circularAdjacency(M_SQRT2), *left = NULL, *right = NULL;
    if (!adj || !heap) goto err;

    left = leftSide(adj, 1.0);
    right = rightSide(adj, 1.0);
    if (!left || !right) goto err;

    costs_ptr[src] = 0;
    preds_ptr[src] = -1;
    labels_ptr[src] = false;

    insertHeap(heap, src, -1);

    Set *seen = NULL;
    pushSet(&seen, src);

    while (!emptyHeap(heap))
    {
        int p = popHeap(heap);

        if (p == dst)
            break;

        Coord u = indexToCoord(&shape, p);
        for (int i = 1; i < adj->size; i++)
        {
            Coord v = adjacentCoord(&u, adj, i);
            if (validCoord(&shape, &v))
            {
                int q = coordToIndex(&shape, &v);
                if (costs_ptr[p] < costs_ptr[q])
                {
                    bool at_border = false;

                    int l = p;
                    Coord ll = adjacentCoord(&u, left, i);
                    if (validCoord(&shape, &ll))
                        l = coordToIndex(&shape, &ll);
                    else
                        at_border = true;

                    int r = p;
                    Coord rr = adjacentCoord(&u, right, i);
                    if (validCoord(&shape, &rr))
                        r = coordToIndex(&shape, &rr);
                    else
                        at_border = true;
                    
                    // avoid crossing existing path
                    if (!labels_ptr[l] || !labels_ptr[r])
                    {
                        double dist = 0.0;
                        if (at_border)
                            dist = 0.1; // hard-coded value
                        else {
                            dist = distance(image_ptr, l, r, d);
                            switch (livewire_fun)
                            {
                                case EXP:
                                    dist = exp( - dist / param1);
                                    break;

                                case EXPSAL:
                                    dist = exp( - dist / param1);
                                    double sal_dist = distance(saliency_ptr, p, q, sal_d);
                                    dist *= exp( - sal_dist / param2);
                                    break;
                            }
                        }

                        dist += costs_ptr[p];

                        if (dist < costs_ptr[q])
                        {
                            preds_ptr[q] = p;
                            costs_ptr[q] = dist;
                            if (heap->colors[q] == WHITE) {
                                insertHeap(heap, q, p);
                                pushSet(&seen, q);
                            } else
                                goUpHeap(heap, q, p);
                        }
                    }
                }
            }
        }
    }
    
    if (!seen) {
        PyErr_SetString(PyExc_TypeError, "Some error occurred in `liveWire` optimum path computation");
        goto err;
    }

    Set *opt_path = NULL;
    if (preds_ptr[dst] != -1) {
        for (int p = dst; p != -1; p = preds_ptr[p])
            pushSet(&opt_path, p);
    }
    
    for (Set *s = seen; s; s = s->next)
    {
        int p = s->value;
        costs_ptr[p] = DBL_MAX;
        preds_ptr[p] = -1;
        labels_ptr[p] = false;
    }

    for (Set *s = opt_path; s; s = s->next)
    {
        int p = s->value;
        costs_ptr[p] = 0;
        labels_ptr[p] = true;
    }

    destroySet(&seen);
    destroyAdjacency(&right);
    destroyAdjacency(&left);
    destroyAdjacency(&adj);
    destroyHeap(&heap);

    if (!opt_path) {
        // leaves function
        Py_RETURN_NONE;
    }

    int length = lengthSet(opt_path);
    npy_intp dim[1] = {length - 1};
    PyObject *path = PyArray_SimpleNew(1, dim, NPY_LONG);
    if (!path)
       PyErr_NoMemory();
    else if (length > 1) {
        long *path_ptr = PyArray_DATA((PyArrayObject*) path);
        int i = length - 1;
        // last node not include
        for (Set *s = opt_path; s->next; s = s->next)
           path_ptr[--i] = s->value;
    }
    
    destroySet(&opt_path);
    return path;

    err:
    PyErr_NoMemory();
    destroyAdjacency(&right);
    destroyAdjacency(&left);
    destroyAdjacency(&adj);
    destroyHeap(&heap);

    return NULL;
}

