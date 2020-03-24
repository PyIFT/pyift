
#ifndef IFT_HEAP_H
#define IFT_HEAP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>


typedef enum {
    WHITE = 0,
    GRAY = 1,
    BLACK = 2,
} nodeColor;


typedef enum {
    MINIMUM = 0,
    MAXIMUM =1,
} removalPolicy;


typedef struct ift_heap {
    double *values;
    int *nodes;
    int *pos;
    int last;
    int size;
    nodeColor  *colors;
    removalPolicy policy;
} Heap;


Heap *createHeap(int size, double *values);
void destroyHeap(Heap **heap_address);
bool insertHeap(Heap *heap, int index);
int popHeap(Heap *heap);
bool removeHeap(Heap *heap, int index);
void resetHeap(Heap *heap);
void goUpHeap(Heap *heap, int index);
void goDownHeap(Heap *heap, int index);


inline bool fullHeap(const Heap *heap)
{
    return (heap->last == (heap->size - 1));
}


inline bool emptyHeap(const Heap *heap)
{
    return (heap->last == -1);
}


#ifdef __cplusplus
}
#endif

#endif // IFT_HEAP_H
