//
//  world.cpp
//  
//
//  Created by ekandrot on 9/23/15.
//
//

#include "maths.h"
#include "world.h"
#include <math.h>
//#include <limits>       // std::numeric_limits


void World::add_sphere(Coord3 center, float r, Color4 c) {
    Sphere s;
    s.center = center;
    s.r = r;
    s.color = c;
    objs.push_back(s);
}

//------------------------------------------------------------------------------

__device__ Color4 cast_ray(const Ray3& ray, Sphere *spheres, int count) {
    Coord3 light = unit_vector({0.35,0.35,1});
    float tmin = 0;
    float tmax = 3.402823466e+38f;//std::numeric_limits<float>::max()numeric_limits;
    float tcurrent = tmax;
    Coord3 normal;
    Coord3 where;
    Color4 returnColor;
    bool hit(false);
    for (int i=0; i<count; ++i) {
        Sphere &s = spheres[i];
        Coord3 temp = ray.origin - s.center;
        float a = dot(ray.direction, ray.direction);
        float b = 2*dot(ray.direction, temp);
        float c = dot(temp, temp) - sqr(s.r);
        
        float discriminant = b*b-4*a*c;
        if (discriminant >= 0) {
            discriminant = sqrt(discriminant);
            float t = (-b - discriminant) / (2*a);
            if (t < tmin) {
                t = (-b + discriminant) / (2*a);
            }
            if (t >= tmin && t <= tmax && t < tcurrent) {
                tcurrent = t;
                returnColor = s.color;
                where = ray.origin + t * ray.direction;
                normal = unit_vector(where - s.center);
                hit = true;
            }
        }
    }
    if (!hit) {
        return returnColor;
    }
    float phong = dot(normal, unit_vector(light-where));
    if (phong <=0) {
        return Color4();
    }
    return phong*returnColor;
}
