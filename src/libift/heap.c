#include "heap.h"
#include <stdlib.h>
#include <float.h>


Heap *createHeap(int size, double *values)
{
    Heap *heap = malloc(sizeof *heap);
    if (!heap) return NULL;
    
    heap->values = values;
    heap->nodes = malloc(size * sizeof *heap->nodes);
    heap->pos = malloc(size * sizeof *heap->pos);
    heap->last = -1;
    heap->size = size;
    heap->colors = malloc(size * sizeof *heap->colors);
    heap->policy = MINIMUM;

    if (!heap->nodes || !heap->pos || !heap->colors)
    {
        if (heap->nodes) free(heap->nodes);
        if (heap->pos) free(heap->pos);
        if (heap->colors) free(heap->colors);
        free(heap);
        return NULL;
    }

    for (int i = 0; i < heap->size; i++)
    {
        heap->nodes[i] = -1;
        heap->pos[i] = -1;
        heap->colors[i] = WHITE;
    }

    return heap;
}


void destroyHeap(Heap **heap_address)
{
    Heap *heap = *heap_address;
    if (!heap) return;
    free(heap->nodes);
    free(heap->pos);
    free(heap->colors);
    free(heap);
    *heap_address = NULL;
}


inline int _dad(int i)
{
    return ((i - 1) / 2);
}


inline int _leftSon(int i)
{
    return (2 * i + 1);
}


inline int _rightSon(int i)
{
    return (2 * i + 2);
}


inline void _swap(Heap *heap, int i, int j)
{
    int tmp = heap->nodes[j];
    heap->nodes[j] = heap->nodes[i];
    heap->nodes[i] = tmp;
    heap->pos[heap->nodes[i]] = i;
    heap->pos[heap->nodes[j]] = j;
}


inline bool _lower(Heap *heap, int i, int j)
{
    return (heap->values[heap->nodes[i]] < heap->values[heap->nodes[j]]);
}


inline bool _greater(Heap *heap, int i, int j)
{
    return (heap->values[heap->nodes[i]] > heap->values[heap->nodes[j]]);
}


void _goUpHeapPosition(Heap *heap, int pos)
{
    int dad = _dad(pos);
    if (heap->policy == MINIMUM) {
        while ((dad >= 0) && _greater(heap, dad, pos))
        {
            _swap(heap, dad, pos);
            pos = dad;
            dad = _dad(pos);
        }
    } else {
        while ((dad >= 0) && _lower(heap, dad, pos))
        {
            _swap(heap, dad, pos);
            pos = dad;
            dad = _dad(pos);
        }
    }
}


void _goDownHeapPosition(Heap *heap, int pos)
{
    int next = pos, left = _leftSon(pos), right = _rightSon(pos);
    if (heap->policy == MINIMUM) {
        if ((left <= heap->last) && _lower(heap, left, next))
            next = left;
        if ((right <= heap->last) && _lower(heap, right, next))
            next = right;
    } else {
        if ((left <= heap->last) && _greater(heap, left, next))
            next = left;
        if ((right <= heap->last) && _greater(heap, right, next))
            next = right;
    }
    if (next != pos) {
        _swap(heap, next, pos);
        _goDownHeapPosition(heap, next);
    }
}


bool insertHeap(Heap *heap, int index)
{
    if (fullHeap(heap))
        return false;
    
    heap->last++;
    heap->nodes[heap->last] = index;
    heap->colors[index] = GRAY;
    heap->pos[index] = heap->last;
    _goUpHeapPosition(heap, heap->last);
    return true;
}


int popHeap(Heap *heap)
{
    int index = -1;
    if (!emptyHeap(heap))
    {
        index = heap->nodes[0];
        heap->pos[index] = -1;
        heap->colors[index] = BLACK;
        heap->nodes[0] = heap->nodes[heap->last];
        heap->pos[heap->nodes[0]] = 0;
        heap->nodes[heap->last] = -1;
        heap->last--;
        _goDownHeapPosition(heap, 0);
    }

    return index;
}


bool removeHeap(Heap *heap, int index)
{
    if (heap->pos[index] == -1)
        return false;
    
    double value = heap->values[index];
    if (heap->policy == MINIMUM)
        heap->values[index] = DBL_MAX;
    else
        heap->values[index] = DBL_MIN;

    goUpHeap(heap, index);
    popHeap(heap);

    heap->values[index] = value;
    heap->colors[index] = WHITE;
    return true;
}


void resetHeap(Heap *heap)
{
    for (int i = 0; i < heap->size; i++)
    {
        heap->nodes[i] = -1;
        heap->pos[i] = -1;
        heap->colors[i] = WHITE;
    }
    heap->last = -1;
}


void goUpHeap(Heap *heap, int index)
{
    _goUpHeapPosition(heap, heap->pos[index]);
}


void goDownHeap(Heap *heap, int index)
{
    _goDownHeapPosition(heap, heap->pos[index]);
}
