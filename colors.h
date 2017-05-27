//
//  colors.h
//
//  Created by ekandrot on 5/26/17.
//
//

#pragma once

struct Color4 {
    __host__ __device__ Color4() : r(0), g(0), b(0), a(1) {}
    __host__ __device__ Color4(float _r, float _g, float _b, float _a) : r(_r), g(_g), b(_b), a(_a) {}
    inline __host__ __device__ Color4& operator+=(const Color4& rhs) {
        r+=rhs.r; g+=rhs.g; b+=rhs.b;
        return *this;
    }
    float r, g, b, a;
};
inline __host__ __device__ Color4 operator+(const Color4& a, const Color4& b) {
    return {a.r+b.r, a.g+b.g, a.b+b.b, 1};
}
inline __host__ __device__ Color4 operator*(float a, const Color4& b) {
    return Color4(a*b.r, a*b.g, a*b.b, 1);
}
inline __host__ __device__ Color4 operator/(const Color4& b, float a) {
    return (1.0/a)*b;
}

