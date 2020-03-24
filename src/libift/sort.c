#include "sort.h"


inline void _swap(float *values, int *indices, int i, int j)
{
    int id = indices[i];
    indices[i] = indices[j];
    indices[j] = id;
    float v = values[i];
    values[i] = values[j];
    values[j] = v;
}


void argSortFloat(float *values, int *indices, int i, int j, sortOrder order)
{
    int m = i;
    if (i < j) {
        int d = (i + j) / 2; // pivot
        _swap(values, indices, d, i);
        if (order == INCREASING) {
            for (d = i + 1; d <= j; d++) {
                if (values[d] < values[i]) {
                    m++;
                    _swap(values, indices, d, m);
                }
            }
        } else {
            for (d = i + 1; d <= j; d++) {
                if (values[d] > values[i]) {
                    m++;
                    _swap(values, indices, d, m);
                }
            }
        }
        _swap(values, indices, m, i);
        argSortFloat(values, indices, i, m - 1, order);
        argSortFloat(values, indices, m + 1, j, order);
    }
}