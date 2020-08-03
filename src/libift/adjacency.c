#include "adjacency.h"
#include "sort.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>


void destroyAdjacency(Adjacency **adj_address)
{
    Adjacency *adj = *adj_address;
    if (!adj) return;
    free(adj->dx);
    free(adj->dy);
    free(adj->dz);
    free(adj);
    *adj_address = NULL;
}


Adjacency *_createAdjacency(int size)
{
    Adjacency *adj = malloc(sizeof *adj);
    if (!adj) return NULL;
    
    adj->dx = malloc(size * sizeof *adj->dx);
    adj->dy = malloc(size * sizeof *adj->dy);
    adj->dz = malloc(size * sizeof *adj->dz);
    adj->size = size;
    if (!adj->dx || !adj->dy || !adj->dz)
    {
        if (!adj->dx) free(adj->dx);
        if (!adj->dy) free(adj->dy);
        if (!adj->dz) free(adj->dz);
        free(adj);
        adj = NULL;
    }
    return adj;
}


Adjacency *circularAdjacency(float radii)
{
    int r_min = radii;
    int r_max = radii * radii + 0.5;
    
    int size = 0;
    for (int dy = -r_min; dy <= r_min; dy++) {
        for (int dx = -r_min; dx <= r_min; dx++) {
            if ((dx * dx + dy * dy) <= r_max)
                size++;
        }
    }

    Adjacency *adj = _createAdjacency(size);
    if (!adj) return NULL;
    float *sq_dists = malloc(size * sizeof *sq_dists);
    if (!sq_dists) {
        destroyAdjacency(&adj);
        return NULL;
    }

    int center = 0;
    for (int i = 0, dy = -r_min; dy <= r_min; dy++) {
        for (int dx = -r_min; dx <= r_min; dx++) {
            float d = dx * dx + dy * dy;
            if (d <= r_max)
            {
                adj->dx[i] = dx;
                adj->dy[i] = dy;
                adj->dz[i] = 0;
                sq_dists[i] = d;
                if ((dx == 0) && (dy == 0))
                    center = i;
                i++;
            }
        }
    }
    
    int *indices = malloc(size * sizeof *indices);
    for (int i = 0; i < size; i++)
        indices[i] = i;

    argSortFloat(sq_dists, indices, 0, size - 1, INCREASING);
    Adjacency *sorted = _createAdjacency(size);
   
    for (int i = 0; i < size; i++) {
        int id = indices[i];
        sorted->dx[i] = adj->dx[id];
        sorted->dy[i] = adj->dy[id];
        sorted->dz[i] = 0;
    }

    free(sq_dists);
    destroyAdjacency(&adj);
    return sorted;
}


Adjacency *sphericAdjacency(float radii)
{
    int r_min = radii;
    int r_max = radii * radii + 0.5;
    
    int size = 0;
    for (int dz = -r_min; dz <= r_min; dz++) {
        for (int dy = -r_min; dy <= r_min; dy++) {
            for (int dx = -r_min; dx <= r_min; dx++) {
                if ((dx * dx + dy * dy + dz * dz) <= r_max)
                    size++;
            }
        }
    }

    Adjacency *adj = _createAdjacency(size);
    if (!adj) return NULL;
    float *sq_dists = malloc(size * sizeof *sq_dists);
    if (!sq_dists) {
        destroyAdjacency(&adj);
        return NULL;
    }
    
    int center = 0;
    for (int i = 0, dz = -r_min; dz <= r_min; dz++) {
        for (int dy = -r_min; dy <= r_min; dy++) {
            for (int dx = -r_min; dx <= r_min; dx++) {
                float d = dx * dx + dy * dy + dz * dz;
                if (d <= r_max)
                {
                    adj->dx[i] = dx;
                    adj->dy[i] = dy;
                    adj->dz[i] = dz;
                    sq_dists[i] = d;
                    if ((dx == 0) && (dy == 0) && (dz == 0))
                        center = i;
                    i++;
                }
            }
        }
    }
    
    int *indices = malloc(size * sizeof *indices);
    for (int i = 0; i < size; i++)
        indices[i] = i;

    argSortFloat(sq_dists, indices, 0, size - 1, INCREASING);
    Adjacency *sorted = _createAdjacency(size);
   
    for (int i = 0; i < size; i++) {
        int id = indices[i];
        sorted->dx[i] = adj->dx[id];
        sorted->dy[i] = adj->dy[id];
        sorted->dz[i] = adj->dz[id];
    }

    free(sq_dists);
    destroyAdjacency(&adj);
    return sorted;
}


Adjacency *leftSide(Adjacency *adj, double shift)
{
    for (int i = 0; i < adj->size; i++)
        if (adj->dz[i] != 0)
            return NULL;

    Adjacency *left = _createAdjacency(adj->size);
    if (!left) return NULL;

    for (int i = 0; i < left->size; i++)
    {
        left->dx[i] = left->dy[i] = left->dz[i] = 0;
        double length = sqrt(adj->dx[i] * adj->dx[i] + adj->dy[i] * adj->dy[i]);
        if (length >=  FLT_EPSILON) {
            left->dx[i] = round((adj->dx[i] / 2.0 + adj->dy[i] / length) * shift);
            left->dy[i] = round((adj->dy[i] / 2.0 - adj->dx[i] / length) * shift);
        }
    }

    return left;
}


Adjacency *rightSide(Adjacency *adj, double shift)
{
    for (int i = 0; i < adj->size; i++)
        if (adj->dz[i] != 0)
            return NULL;

    Adjacency *right = _createAdjacency(adj->size);
    if (!right) return NULL;

    for (int i = 0; i < right->size; i++)
    {
        right->dx[i] = right->dy[i] = right->dz[i] = 0;
        double length = sqrt(adj->dx[i] * adj->dx[i] + adj->dy[i] * adj->dy[i]);
        if (length >=  FLT_EPSILON) {
            right->dx[i] = round((adj->dx[i] / 2.0 - adj->dy[i] / length) * shift);
            right->dy[i] = round((adj->dy[i] / 2.0 + adj->dx[i] / length) * shift);
        }
    }

    return right;
}
