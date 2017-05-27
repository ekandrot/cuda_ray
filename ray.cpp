#include <cstddef>
#include <iostream>
#include "png_wrapper.h"
#include "coords.h"
#include "colors.h"
#include "world.h"

#define __DRIVER_TYPES_H__
#include "helper_cuda.h"

const int W = 1024;
const int H = 1024;

//------------------------------------------------------------------------------

// inclusive
static int rand_range(int a, int b) {
    return a + rand() % (b-a+1);
}

//------------------------------------------------------------------------------

static __device__ Ray3 screen_to_ray(float x, float y, float o) {
    Ray3 ray;
    ray.origin.x = o/W;
    ray.origin.y = 0;
    ray.origin.z = 5;
    Coord3 lookat;
    lookat.x = (2*(x/W) - 1);
    lookat.y = (1 - 2*(y/H));
    lookat.z = 0;
    ray.direction = unit_vector(lookat - ray.origin);
    return ray;
}

//------------------------------------------------------------------------------

static uint8_t srgbEncode(float c) {
    float x = c;
    if (c <= 0.0031308f) {
        x = 12.92f * c;
    } else {
        x = 1.055f * pow(c, 1/2.4) - 0.055f;
    }
    x *= 256;
    if (x<0) {
        x = 0;
    } else if (x>255) {
        x = 255;
    }
    return x;
}

//------------------------------------------------------------------------------

__global__ void ray_kernel(float *image, Sphere *spheres, int count) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    Ray3 r = screen_to_ray(x, y, 0);
    Color4 c = cast_ray(r, spheres, count);
    image[4*(x+y*W)+0] = c.r;
    image[4*(x+y*W)+1] = c.g;
    image[4*(x+y*W)+2] = c.b;
    image[4*(x+y*W)+3] = c.a;
}

//------------------------------------------------------------------------------

void gen_image_cuda(int w, int h, float *image, const World &world) {
    const size_t imageMemSize = (w*h*4*sizeof(float));
    float *d_image;
    Sphere *d_s;
    checkCudaErrors(cudaMalloc((void**)&d_image, imageMemSize));
    checkCudaErrors(cudaMalloc((void**)&d_s, sizeof(Sphere)*world.objs.size()));
    checkCudaErrors(cudaMemcpy(d_s, world.objs.data(), sizeof(Sphere)*world.objs.size(), cudaMemcpyHostToDevice));

    dim3 t(16,16,1);
    dim3 b(w/16,h/16,1);
    ray_kernel<<<b,t>>>(d_image, d_s, world.objs.size());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(image, d_image, imageMemSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_image));
    checkCudaErrors(cudaFree(d_s));
}

//------------------------------------------------------------------------------

void gen_image(int w, int h, float *image) {
    for (int y=0; y<h; ++y) {
        for (int x=0; x<w; ++x) {
            image[4*(x+y*w)] = 1.0f*x/w;
        } 
    }
}

//------------------------------------------------------------------------------

int main(int argc, char **argv) {

    World world;

//    world.add_sphere({0,0,-1}, 0.4, {1,0,0,1});
    for (int i=0; i<100; ++i) {
        world.add_sphere(Coord3(rand_range(-10,10)/10.0f, rand_range(-10,10)/10.0f, -rand_range(-10,10)/10.0f),
                     rand_range(1,10) / 30.0f,
                     {rand_range(0,10)/10.0f, rand_range(0,10)/10.0f, rand_range(0,10)/10.0f, 1});
    }



    float *image = (float*)malloc(W*H*4*sizeof(float)); // rgba
    gen_image_cuda(W, H, image, world);
//    gen_image(W, H, image);

    RGBBitmap   bitmap;
    bitmap.width = 1024;
    bitmap.height = 1024;
    bitmap.bytewidth = 3*1024;
    bitmap.bytes_per_pixel = 3;
    bitmap.pixels = (RGBPixel*)malloc(W*H*3);

    for (int i=0; i<W*H; ++i) {
        bitmap.pixels[i].red = srgbEncode(image[4*i+0]);
        bitmap.pixels[i].green = srgbEncode(image[4*i+1]);
        bitmap.pixels[i].blue = srgbEncode(image[4*i+2]);
    }

    save_png_to_file(&bitmap, "xxx.png");

    free(bitmap.pixels);
    free(image);

    return 0;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

