#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#include "matrix.h"
#include <time.h>
#define TWOPI 6.2831853

// Calculate image derivatives Ix and Iy.
// Calculate measures Ix^2, Iy^2, and Ix * Iy.
// Calculate structure matrix S as weighted sum of nearby measures.
// Calculate Harris "cornerness" as estimate of 2nd eigenvalue: det(S) - α trace(S)^2, α = .06
// Run non-max suppression on response map

// Frees an array of descriptors.
// descriptor *d: the array.
// int n: number of elements in array.
void free_descriptors(descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(d[i].data);
    }
    free(d);
}

// Create a feature descriptor for an index in an image.
// image im: source image.
// int i: index in image for the pixel we want to describe.
// returns: descriptor for that index.
descriptor describe_index(image im, int i)
{
    int w = 5;
    descriptor d;
    d.p.x = i%im.w;
    d.p.y = i/im.w;
    d.data = calloc(w*w*im.c, sizeof(float));
    d.n = w*w*im.c;
    int c, dx, dy;
    int count = 0;
    // If you want you can experiment with other descriptors
    // This subtracts the central value from neighbors
    // to compensate some for exposure/lighting changes.
    for(c = 0; c < im.c; ++c){
        float cval = im.data[c*im.w*im.h + i];
        for(dx = -w/2; dx < (w+1)/2; ++dx){
            for(dy = -w/2; dy < (w+1)/2; ++dy){
                float val = get_pixel(im, i%im.w+dx, i/im.w+dy, c);
                d.data[count++] = cval - val;
            }
        }
    }
    return d;
}

// Marks the spot of a point in an image.
// image im: image to mark.
// ponit p: spot to mark in the image.
void mark_spot(image im, point p)
{
    int x = p.x;
    int y = p.y;
    int i;
    for(i = -9; i < 10; ++i){
        set_pixel(im, x+i, y, 0, 1);
        set_pixel(im, x, y+i, 0, 1);
        set_pixel(im, x+i, y, 1, 0);
        set_pixel(im, x, y+i, 1, 0);
        set_pixel(im, x+i, y, 2, 1);
        set_pixel(im, x, y+i, 2, 1);
    }
}

// Marks corners denoted by an array of descriptors.
// image im: image to mark.
// descriptor *d: corners in the image.
// int n: number of descriptors to mark.
void mark_corners(image im, descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        mark_spot(im, d[i].p);
    }
}

// Creates a 1d Gaussian filter.
// float sigma: standard deviation of Gaussian.
// returns: single row image of the filter.
image make_1d_gaussian(float sigma)
{
    int dim = ceil(6*sigma);
    dim += ((dim % 2) ? 0 : 1);
    int c = dim / 2;
    image filter = make_image(dim, 1, 1);

    for (int x = 0; x < filter.w; x++) {
            filter.data[x] = 1 / (TWOPI * sigma * sigma) *
                               exp(-((x-c)*(x-c) + (x-c)*(x-c)) / (2*sigma*sigma));
    }

    l1_normalize(filter);

    return filter;
}

// Smooths an image using separable Gaussian filter.
// image im: image to smooth.
// float sigma: std dev. for Gaussian.
// returns: smoothed image.
image smooth_image(image im, float sigma)
{
    int x;
    image gauss1,gauss2;

    gauss1 = make_1d_gaussian(sigma);
    gauss2 = make_image(1, gauss1.w, 1);
    for (x = 0; x < gauss1.w; x++) {
        gauss2.data[x] = gauss1.data[x];
    }

    im = convolve_image(im, gauss1, 1);
    im = convolve_image(im, gauss2, 1);

    return im;
}

// Calculate the structure matrix of an image.
// image im: the input image.
// float sigma: std dev. to use for weighted sum.
// returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
//          third channel is IxIy.
image structure_matrix(image im, float sigma)
{
    // This will perform the first 3 steps of the algorithm: calculating derivatives, the corresponding measures, and the weighted sum of nearby derivative information.
    // You can use Sobel filter and associated functions from HW2 to calculate the derivatives.
    // The measures are element-wise multiplications.
    // The weighted sum can be easily computed with a Gaussian blur as discussed in class.
    // Use the parameter sigma to create the Gaussian kernel and convolve the result with it.
    int pos;
    image ret = make_image(im.w, im.h, 3);

    image gx = make_gx_filter();
    image gy = make_gy_filter();
    image Ix = convolve_image(im, gx, 0);
    image Iy = convolve_image(im, gy, 0);

    // Calculate measures Ix^2, Iy^2, and Ix * Iy.
    for (int x = 0; x < im.w; x++) {
    for (int y = 0; y < im.h; y++) {
            pos = x + y*im.w;
            ret.data[pos + 0*im.w*im.h] = Ix.data[pos] * Ix.data[pos];
            ret.data[pos + 1*im.w*im.h] = Iy.data[pos] * Iy.data[pos];
            ret.data[pos + 2*im.w*im.h] = Ix.data[pos] * Iy.data[pos];
    }
    }

    image gauss = make_gaussian_filter(sigma);
    ret = convolve_image(ret, gauss, 1);

    return ret;
}

// Estimate the cornerness of each pixel given a structure matrix S.
// image S: structure matrix for an image.
// returns: a response map of cornerness calculations.
image cornerness_response(image S)
{
    // cornerness = det(S) - α trace(S)^2, α = .06
    // det(S) = λ_1 * λ_2 = ad-bc
    // trace(S) = a+d = λ_1 + λ_2

    int pos;
    float lam_1, lam_2;
    float a, b, c, d;
    float trace, det;
    float alpha = 0.06;
    image R = make_image(S.w, S.h, 1);

    for (int x = 0; x < S.w; x++) {
    for (int y = 0; y < S.h; y++) {
	    pos = x + y * S.w;
	    a = S.data[pos + 0 * S.w * S.h]; // Ix^2
	    b = S.data[pos + 2 * S.w * S.h]; // IxIy
	    c = S.data[pos + 2 * S.w * S.h]; // IyIx
	    d = S.data[pos + 1 * S.w * S.h]; // Iy^2
	    lam_1 = 1/2.0 * ((a+d) + sqrtf(4*b*c + (a-d)*(a-d)));
	    lam_2 = 1/2.0 * ((a+d) - sqrtf(4*b*c + (a-d)*(a-d)));
	    trace = lam_1 + lam_2;
	    det = lam_1 * lam_2;
	    R.data[pos] = det - alpha*trace*trace;
    }
    }

    return R;
}

// Perform non-max supression on an image of feature responses.
// image im: 1-channel image of feature responses.
// int w: distance to look for larger responses.
// returns: image with only local-maxima responses within w pixels.
image nms_image(image im, int w)
{

    // For every pixel in im, check every neighbor within w pixels (Chebyshev distance).
    // Equivalently, check the 2w+1 window centered at each pixel.
    // If any responses are stronger, suppress that pixel's response (set it to a very low negative number).

    image R = copy_image(im);
    for (int x = 0; x < im.w; x++) {
    for (int y = 0; y < im.h; y++) {
	for (int offset_x = -w; offset_x <= w; offset_x++) {
	for (int offset_y = -w; offset_y <= w; offset_y++) {
	    if (get_pixel(im, x+offset_x, y+offset_y, 0) > get_pixel(im, x, y, 0)) {
		set_pixel(R, x, y, 0, -9999.9);
	    }
	}
	}
    }
    }

    return R;
}

// Perform harris corner detection and extract features from the corners.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
// int *n: pointer to number of corners detected, should fill in.
// returns: array of descriptors of the corners in the image.
descriptor *harris_corner_detector(image im, float sigma, float thresh, int nms, int *n)
{
    // The function should return an array of descriptors for corners in the image.
    // Code for calculating the descriptors is provided. Also, set the integer *n to be the number of corners found in the image.

    // Calculate structure matrix
    image S = structure_matrix(im, sigma);

    // Estimate cornerness
    image R = cornerness_response(S);

    // Run NMS on the responses
    image Rnms = nms_image(R, nms);


    // TODO: count number of responses over threshold

    // int count = 1; // change this
    int count = 0;
    for (int i = 0; i < (Rnms.w*Rnms.h); i++) {
	if (Rnms.data[i] > thresh) count++;
    }

    // *n = count; // <- set *n equal to number of corners in image.
    // descriptor *d = calloc(count, sizeof(descriptor));
    //TODO: fill in array *d with descriptors of corners, use describe_index.

    *n = count;
    descriptor *d = calloc(count, sizeof(descriptor));
    for (int i = 0, j = 0; i < (Rnms.w*Rnms.h); i++) {
	if (Rnms.data[i] > thresh) {
	    d[j] = describe_index(im, i);
	    if (j == count) break; else j++;
	}
    }

    // Free images
    free_image(S);
    free_image(R);
    free_image(Rnms);

    return d;
}

// Find and draw corners on an image.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
void detect_and_draw_corners(image im, float sigma, float thresh, int nms)
{
    int n = 0;
    descriptor *d = harris_corner_detector(im, sigma, thresh, nms, &n);
    mark_corners(im, d, n);
}
