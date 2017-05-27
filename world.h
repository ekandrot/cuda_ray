//
//  world.h
//  
//  Created by ekandrot on 5/26/17.
//
//

#pragma once

#include "coords.h"
#include "colors.h"
#include <stdio.h>
#include <math.h>
#include <vector>
#include <thrust/device_vector.h>

struct Ray3 {
    Coord3 origin;
    Coord3 direction;
};


struct Sphere {
    Coord3 center;
    float r;
    Color4 color;
};


struct World {
    void add_sphere(Coord3 center, float r, Color4 c);
    
//private:
    std::vector<Sphere> objs;
};

__device__ Color4 cast_ray(const Ray3& ray, Sphere *spheres, int count);

