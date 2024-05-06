#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

/******************************** Resizing *****************************
  To resize we'll need some interpolation methods and a function to create
  a new image and fill it in with our interpolation methods.
************************************************************************/

float nn_interpolate(image im, float x, float y, int c)
{
    // TODO
    /***********************************************************************
      This function performs nearest-neighbor interpolation on image "im"
      given a floating column value "x", row value "y" and integer channel "c",
      and returns the interpolated value.
    ************************************************************************/
    // Remember to use the closest int, not just type-cast because in C that will truncate towards zero.
    // NOTE: your rounding of x and y should account for the fact that x and/or y may be integers.

    int new_x, new_y;

    new_x = round(x);
    new_y = round(y);
    new_x = fmin(fmax(0,new_x),im.w-1);
    new_y = fmin(fmax(0,new_y),im.h-1);

    return im.data[new_x + new_y * im.w + c * im.w * im.h];
}

image nn_resize(image im, int w, int h)
{
    // TODO Fill in (also fix the return line)
    /***********************************************************************
      This function uses nearest-neighbor interpolation on image "im" to a new
      image of size "w x h"
    ************************************************************************/
    // Create a new image that is w x h and the same number of channels as im
    // Loop over the pixels and map back to the old coordinates (remember to use 0.5 offset appropriately)
    // Use nn_interpolate() to fill in the image
    // try on python

    int x, y, ch, pos;
    float a, b, c, d;
    float old_x, old_y;
    image resize = make_image(w, h, im.c);

    a = (float)im.w / w;
    b = 0.5 * a - 0.5;
    c = (float)im.h / h;
    d = 0.5 * c - 0.5;
    for (ch = 0; ch < im.c; ch++) {
      for (x = 0; x < w; x++) {
        for (y = 0; y < h; y++) {
          old_x = a * x + b;
          old_y = c * y + d;
          pos = x + y * w + ch * w * h;
          resize.data[pos] = nn_interpolate(im, old_x, old_y, ch);
        }
      }
    }
    return resize;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    // TODO
    /***********************************************************************
      This function performs bilinear interpolation on image "im" given
      a floating column value "x", row value "y" and integer channel "c".
      It interpolates and returns the interpolated value.
    ************************************************************************/
    // Fill in the function float bilinear_interpolate(image im, float x, float y, int c) for bilinear interpolation.

    int x_min = floor(x);
    int x_max = ceil(x);
    int y_min = floor(y);
    int y_max = ceil(y);

    float d1 = x - x_min;
    float d2 = x_max - x;
    float d3 = y - y_min;
    float d4 = y_max - y;

    float V1 = get_pixel(im, x_min, y_min, c);
    float V2 = get_pixel(im, x_max, y_min, c);
    float V3 = get_pixel(im, x_min, y_max, c);
    float V4 = get_pixel(im, x_max, y_max, c);
    
    float A1 = d2*d4;
    float A2 = d1*d4;
    float A3 = d2*d3;
    float A4 = d1*d3;
    float Q = V1*A1 + V2*A2 + V3*A3 + V4*A4;

    return Q;
}

image bilinear_resize(image im, int w, int h)
{
    // TODO
    /***********************************************************************
      This function uses bilinear interpolation on image "im" to a new image
      of size "w x h". Algorithm is same as nearest-neighbor interpolation.
        ************************************************************************/
    // Fill in image bilinear_resize(image im, int w, int h) to perform resizing using bilinear interpolation. Try it out again in python

    int x, y, ch, pos;
    float a, b, c, d;
    float old_x, old_y;
    image resize = make_image(w, h, im.c);

    a = (float)im.w / w;
    b = 0.5 * a - 0.5;
    c = (float)im.h / h;
    d = 0.5 * c - 0.5;
    for (ch = 0; ch < im.c; ch++) {
        for (x = 0; x < w; x++) {
        for (y = 0; y < h; y++) {
            old_x = a * x + b;
            old_y = c * y + d;
            pos = x + y * w + ch * w * h;
            resize.data[pos] = bilinear_interpolate(im, old_x, old_y, ch);
        }
        }
    }
    return resize;
}


/********************** Filtering: Box filter ***************************
  We want to create a box filter. We will only use square box filters.
************************************************************************/

void l1_normalize(image im)
{
    // TODO
    /***********************************************************************
      This function divides each value in image "im" by the sum of all the
      values in the image and modifies the image in place.
    ************************************************************************/
    // This should divide each value in image im by the sum of all the values in the image.
    float sum = 0;
    for (int i = 0; i < (im.c * im.h * im.w); i++) {
        sum += im.data[i];
    }
    for (int i = 0; i < (im.c * im.h * im.w); i++) {
        im.data[i] = im.data[i] / sum;
    }
}

image make_box_filter(int w)
{
    // TODO
    /***********************************************************************
      This function makes a square filter of size "w x w". Make an image of
      width = height = w and number of channels = 1, with all entries equal
      to 1. Then use "l1_normalize" to normalize your filter.
    ************************************************************************/
    // We will only use square box filters, so just create a square image of width = height = w and number of channels = 1, with all entries equal to 1.
    // Then use l1_normalize to normalize your filter.
    image box = make_image(w,w,1);

    for (int i=0; i < (box.h*box.w); i++) {
        box.data[i] = 1;
    }

    l1_normalize(box);
    return box;
}

image convolve_image(image im, image filter, int preserve)
{
    // TODO
    /***********************************************************************
      This function convolves the image "im" with the "filter". The value
      of preserve is 1 if the number of input image channels need to be 
      preserved. Check the detailed algorithm given in the README.  
    ************************************************************************/
    // The parameter =preserve= takes a value of either 0 or 1.
    // We will use "clamp" padding for the image borders (get_pixel() already handles it).
    // For this function we have a few scenarios. With normal convolutions we do a weighted sum over an area of the image. With multiple channels in the input image there are a few possible cases we want to handle:

    /*
      If filter and im have the same number of channels then it's just a normal convolution. We sum over spatial and channel dimensions and produce a 1 channel image.
      UNLESS:
      If preserve is set to 1 we should produce an image with the same number of channels as the input. This is useful if, for example, we want to run a box filter over an RGB image and get out an RGB image. This means each channel in the image will be filtered by the corresponding channel in the filter. UNLESS:
      If the filter only has one channel but im has multiple channels we want to apply the filter to each of those channels. Then we either sum between channels or not depending on if preserve is set.
    */

    // Also, filter should have the same number of channels as im or have 1 channel. This MUST be checked with an assert(). Hint: You can reduce number of lines of code by using the conditional operator in C.
    // We are calling this a convolution but you don't need to flip the filter or anything (we're actually doing a cross-correlation). Just apply it to the image as we discussed in class:

    assert(im.c == filter.c || filter.c == 1);

    image result = make_image(im.w,im.h, preserve ? im.c : 1);
    int offsetx = floor(filter.w/2);
    int offsety = floor(filter.h/2);

    for (int x = 0; x < im.w; x++) {
    for (int y = 0; y < im.h; y++) {
    for (int c = 0; c < im.c; c++) {
        int pos = x + y * im.w + (preserve ? c : 0) * im.w * im.h;
        for (int yf = -offsety; yf <= offsety; yf++) {
        for (int xf = -offsetx; xf <= offsetx; xf++) {
            int posf = (xf + offsetx) + (yf + offsety) * filter.w;
            result.data[pos] += filter.data[posf] * get_pixel(im,x+xf,y+yf, c);
        }
        }
    }
    }
    }
    return result;

}

image make_highpass_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 filter with highpass filter values using image.data[]
    ************************************************************************/

    // Fill in the functions image make_highpass_filter(), image make_sharpen_filter(), and image make_emboss_filter() to return the example kernels we covered in class.
    // Answer Questions 2.3.1 and 2.3.2 in the source file (put your answers just right there).

    image filter = make_image(3,3,1);
    float arr[] = {0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0};
    memcpy(filter.data, arr, sizeof(arr));

    return filter;
}

image make_sharpen_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 filter with sharpen filter values using image.data[]
    ************************************************************************/
    image filter = make_image(3,3,1);
    
    filter.data[0] = 0;
    filter.data[1] = -1;
    filter.data[2] = 0;
    filter.data[3] = -1;
    filter.data[4] = 5;
    filter.data[5] = -1;
    filter.data[6] = 0;
    filter.data[7] = -1;
    filter.data[8] = 0;

    return filter;
}

image make_emboss_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 filter with emboss filter values using image.data[]
    ************************************************************************/
    image filter = make_image(3,3,1);
    
    filter.data[0] = -2;
    filter.data[1] = -1;
    filter.data[2] = 0;
    filter.data[3] = -1;
    filter.data[4] = 1;
    filter.data[5] = 1;
    filter.data[6] = 0;
    filter.data[7] = 1;
    filter.data[8] = 2;

    return filter;
}

// Question 2.3.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Answer: TODO

// Question 2.3.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: TODO

image make_gaussian_filter(float sigma)
{
    // TODO
    /***********************************************************************
      sigma: a float number for the Gaussian.
      Create a Gaussian filter with the given sigma. Note that the kernel size 
      is the next highest odd integer from 6 x sigma. Return the Gaussian filter.
    ************************************************************************/
    // 
    // Fill in image make_gaussian_filter(float sigma) which will take a standard deviation value sigma and return a filter that smooths using a gaussian with that sigma. How big should the filter be? 99% of the probability mass for a gaussian is within +/- 3 standard deviations, so make the kernel be 6 times the size of sigma. But also we want an odd number, so make it be the next highest odd integer from 6 x sigma. We need to fill in our kernel with some values (take care of the 0.5 offset for the pixel co-ordinates). Use the probability density function for a 2D gaussian: 2d gaussian

    // Technically this isn't perfect, what we would really want to do is integrate over the area covered by each cell in the filter. But that's much more complicated and this is a decent estimate. Remember though, this is a blurring filter so we want all the weights to sum to 1 (i.e. normalize the filter). Now you should be able to try out your new blurring function:

    int pos;
    int dim = ceil(6*sigma);
    dim += ((dim % 2) ? 0 : 1);
    image filter = make_image(dim, dim, 1);
    int c = dim / 2;

    for (int x = 0; x < filter.w; x++) {
    for (int y = 0; y < filter.h; y++) {
            pos = x + y * filter.w;
            filter.data[pos] = 1 / (TWOPI * sigma * sigma) *
                               exp(-((x-c)*(x-c) + (y-c)*(y-c)) / (2*sigma*sigma));
    }
    }

    l1_normalize(filter);

    return filter;
}

image add_image(image a, image b)
{
    // TODO
    /***********************************************************************
      The input images a and image b have the same height, width, and channels.
      Sum the given two images and return the result, which should also have
      the same height, width, and channels as the inputs. Do necessary checks.
    ************************************************************************/
    // Fill in image add_image(image a, image b) to add two images a and b and image sub_image(image a, image b) to subtract image b from image a, so that we can perform our transformations of + and - like this:
    assert(a.w == b.w && a.h == b.h && a.c == b.c);

    int pos;
    image result = make_image(a.w, a.h, a.c);

    for (int x = 0; x < result.w; x++) {
    for (int y = 0; y < result.h; y++) {
    for (int c = 0; c < result.c; c++) {
        pos = x + y*result.w + c*result.w*result.h;
        result.data[pos] = a.data[pos] + b.data[pos];
    }
    }
    }

    return result;

}

image sub_image(image a, image b)
{
    // TODO
    /***********************************************************************
      The input image a and image b have the same height, width, and channels.
      Subtract the given two images and return the result, which should have
      the same height, width, and channels as the inputs. Do necessary checks.
    ************************************************************************/

    assert(a.w == b.w && a.h == b.h && a.c == b.c);

    int pos; 
    image im = make_image(a.w, a.h, a.c);

    for (int x = 0; x < im.w; x++) {
    for (int y = 0; y < im.h; y++) {
    for (int c = 0; c < im.c; c++) {
      pos = x + y*im.w + c*im.w*im.h;
      im.data[pos] = a.data[pos] - b.data[pos];
    }
    }
    }

    return im;
}

image make_gx_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 Sobel Gx filter and return it
    ************************************************************************/

    image filter = make_image(3,3,1);
    
    filter.data[0] = -1;
    filter.data[1] = 0;
    filter.data[2] = 1;
    filter.data[3] = -2;
    filter.data[4] = 0;
    filter.data[5] = 2;
    filter.data[6] = -1;
    filter.data[7] = 0;
    filter.data[8] = 1;

    return filter;
}

image make_gy_filter()
{
    // TODO
    /***********************************************************************
      Create a 3x3 Sobel Gy filter and return it
    ************************************************************************/
    image filter = make_image(3,3,1);
    
    filter.data[0] = -1;
    filter.data[1] = -2;
    filter.data[2] = -1;
    filter.data[3] = 0;
    filter.data[4] = 0;
    filter.data[5] = 0;
    filter.data[6] = 1;
    filter.data[7] = 2;
    filter.data[8] = 1;

    return filter;
}

void feature_normalize(image im)
{
    // TODO
    /***********************************************************************
      Calculate minimum and maximum pixel values. Normalize the image by
      subtracting the minimum and dividing by the max-min difference.
    ************************************************************************/
  
    float min_val = 0;
    float max_val = 0;

    for (int i = 0; i < (im.c * im.h * im.w); i++) {
        min_val = fmin(min_val,im.data[i]);
        max_val = fmax(max_val,im.data[i]);
    }

    float range = max_val - min_val;

    if (abs(range) < 1e-8) {
        for (int i = 0; i < (im.c * im.h * im.w); i++) {
            im.data[i] = 0;
        }
    } else {
        for (int i = 0; i < (im.c * im.h * im.w); i++) {
            im.data[i] = (im.data[i] - min_val) / range;
        }
    }
}

image *sobel_image(image im)
{
    // TODO
    /***********************************************************************
      Apply Sobel filter to the input image "im", get the magnitude as sobelimg[0]
      and gradient as sobelimg[1], and return the result.
    ************************************************************************/
    // image *sobelimg = calloc(2, sizeof(image));
    // return sobelimg;

    int i;
    image *sobelimg = calloc(2, sizeof(image));
    image mag = make_image(im.w, im.h, 1);
    image dir = make_image(im.w, im.h, 1);

    image gx = convolve_image(im, make_gx_filter(), 0);
    image gy = convolve_image(im, make_gy_filter(), 0);
    for (i; i < (im.w * im.h); i++) {
        mag.data[i] = sqrtf(gx.data[i] * gx.data[i] + gy.data[i] * gy.data[i]);
        dir.data[i] = atan2f(gy.data[i], gx.data[i]);
    }
    sobelimg[0] = mag;
    sobelimg[1] = dir;
    return sobelimg;
}

image colorize_sobel(image im)
{
  // TODO
  /***********************************************************************
    Create a colorized version of the edges in image "im" using the 
    algorithm described in the README.
  ************************************************************************/
  // Write a function image colorize_sobel(image im). Call image *sobel_image(image im), use the magnitude to specify the saturation and value of an image and the angle (direction) to specify the hue. Then use the hsv_to_rgb() function we wrote in Homework 1. The result should look similar to this:

    image ret = make_image(im.w, im.h, 3);
    image *sobelimg = sobel_image(im);
    image mag = sobelimg[0];
    image dir = sobelimg[1];


    for (int x = 0; x < ret.w; x++) {
    for (int y = 0; y < ret.h; y++) {
        ret.data[x+y*ret.w+0*ret.w*ret.h] = dir.data[x+y*ret.w];
        ret.data[x+y*ret.w+1*ret.w*ret.h] = mag.data[x+y*ret.w];
        ret.data[x+y*ret.w+2*ret.w*ret.h] = mag.data[x+y*ret.w];
    }
    }
    hsv_to_rgb(ret);
    return ret;
}

// EXTRA CREDIT: Median filter

int compare_floats(const void* a, const void* b) {
    float arg1 = *(const float*)a;
    float arg2 = *(const float*)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

image apply_median_filter(image im, int kernel_size) {
    int pad = kernel_size / 2;
    image result = make_image(im.w, im.h, im.c);
    float* window = malloc(kernel_size * kernel_size * sizeof(float));

    for (int c = 0; c < im.c; c++) {
    for (int x = 0; x < im.w; x++) {
    for (int y = 0; y < im.h; y++) {
	int k = 0;
	for (int i = -pad; i <= pad; i++) {
	for (int j = -pad; j <= pad; j++) {
	    int nx = x + j;
	    int ny = y + i;
	    window[k++] = get_pixel(im, nx, ny, c);
	}
	}
	qsort(window, k, sizeof(float), compare_floats);
	set_pixel(result, x, y, c, window[k/2]);
    }
    }
    }

    free(window);
    return result;
}

// SUPER EXTRA CREDIT: Bilateral filter

image apply_bilateral_filter(image im, float sigma1, float sigma2) {
    int d = ceil(3 * sigma1);
    image result = make_image(im.w, im.h, im.c);

    for (int c = 0; c < im.c; c++) {
    for (int x = 0; x < im.w; x++) {
    for (int y = 0; y < im.h; y++) {
	float sum = 0.0;
	float sum_weights = 0.0;
	for (int i = -d; i <= d; i++) {
	for (int j = -d; j <= d; j++) {
	    int nx = x + j;
	    int ny = y + i;
	    float dist_sq = i*i + j*j;
	    float s_weight = exp(-dist_sq / (2 * sigma1 * sigma1));
	    float r_diff = get_pixel(im, nx, ny, c) - get_pixel(im, x, y, c);
	    float r_weight = exp(-r_diff * r_diff / (2 * sigma2 * sigma2));
	    float weight = s_weight * r_weight;
	    sum += get_pixel(im, nx, ny, c) * weight;
	    sum_weights += weight;
	}
	}
	set_pixel(result, x, y, c, sum / sum_weights);
    }
    }
    }
    return result;
}

