//
//  coords.h
//
//  Created by ekandrot on 5/26/17.
//
//

#pragma once
#include <math.h>

struct Coord3 {
    __host__ __device__ Coord3() : x(0), y(0), z(0) {}
    __host__ __device__ Coord3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    inline __host__ __device__ Coord3& operator+=(const Coord3& b) {
        x+=b.x; y+=b.y; z+=b.z;
        return *this;
    }
    float x, y, z;
};
inline __host__ __device__ Coord3 operator-(const Coord3& a, const Coord3& b) {
    return {a.x-b.x, a.y-b.y, a.z-b.z};
}
inline __host__ __device__ Coord3 operator+(const Coord3& a, const Coord3& b) {
    return {a.x+b.x, a.y+b.y, a.z+b.z};
}
inline __host__ __device__ Coord3 operator*(float a, const Coord3& b) {
    return {a*b.x, a*b.y, a*b.z};
}
inline __host__ __device__ Coord3 operator/(const Coord3& b, float a) {
    return (1.0/a)*b;
}
inline __host__ __device__ float dot(const Coord3& a, const Coord3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
inline __host__ __device__ Coord3 unit_vector(const Coord3& a) {
    float norm = sqrt(dot(a, a));
    return a/norm;
}
