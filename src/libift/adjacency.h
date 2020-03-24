
#ifndef IFT_ADJACENCY_H
#define IFT_ADJACENCY_H

#ifdef __cplusplus
extern "C" {
#endif

#include "core.h"


typedef struct ift_adjacency
{
    int *dx;
    int *dy;
    int *dz;
    int size;
} Adjacency;


void destroyAdjacency(Adjacency **adj_address);
Adjacency *circularAdjacency(float radii);
Adjacency *sphericAdjacency(float radii);


inline Coord adjacentCoord(const Coord *coord, const Adjacency *adj, int index)
{
    Coord neighbour = {.x = coord->x + adj->dx[index],
                       .y = coord->y + adj->dy[index],
                       .z = coord->z + adj->dz[index]};
    return neighbour;
}



#ifdef __cplusplus
}
#endif

#endif // IFT_ADJACENCY_H