
#ifndef IFT_CORE_H
#define IFT_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdlib.h>


typedef struct ift_coord
{
    int x;
    int y;
    int z;
} Coord;


inline Coord indexToCoord(const Coord *shape, int index)
{
    div_t res1 = div(index, shape->x * shape->y);
    div_t res2 = div(res1.rem, shape->x);
    Coord coord = {.x = res2.rem, .y = res2.quot, .z = res1.quot};
    return coord;
}


inline int coordToIndex(const Coord *shape, const Coord *coord)
{
    return coord->x + // x
           coord->y * shape->x + // y
           coord->z * shape->x * shape->y; // z
}


inline bool validCoord(const Coord *shape, const Coord *coord)
{
    return (coord->x >= 0 && coord->x < shape->x &&
            coord->y >= 0 && coord->y < shape->y && 
            coord->z >= 0 && coord->z < shape->z);
}


#ifdef __cplusplus
}
#endif

#endif // IFT_CORE_H