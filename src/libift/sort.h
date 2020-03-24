
#ifndef IFT_SORT_H
#define IFT_SORT_H

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
    INCREASING,
    DECREASING,
} sortOrder;


void argSortFloat(float *values, int *indices, int first, int last, sortOrder order);


#ifdef __cplusplus
}
#endif

#endif // IFT_SORT_H