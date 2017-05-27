//
//  maths.h
//
//  Created by ekandrot on 5/26/17.
//
//

#pragma once

//#define sqr(x) ((x) * (x))
template <typename T>
inline __device__ __host__ T sqr(T x) {
    return x * x;
}
