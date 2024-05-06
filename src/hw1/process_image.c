#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

// typedef struct{
//     int h,w,c;
//     float *data;
// } image;

float get_pixel(image im, int x, int y, int c)
{
  // return the pixel value at column x, row y, and channel c
  // do bounds checking 
  // we will perform clamp padding to the image
  
    //x + y*W + z*W*H
  if (y < 0) {
    y = 0;
  }
  if (y >= im.h) {
    y = im.h-1;
  }
  if (x < 0) {
    x = 0;
  }
  if (x >= im.w) {
    x = im.w-1;
  }
  if (c >= 0 && c <= im.c) {
    return im.data[x + y*im.w + c*im.w*im.h];
  } else {
    return 0;
  }
}

void set_pixel(image im, int x, int y, int c, float v)
{
    // set the pixed to the value v
    if (y >= 0 && y < im.h &&
	x >= 0 && x < im.w && 
	c >= 0 && c < im.c) {
	im.data[x + y * im.w + c * im.w * im.h] = v;
    }
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    // memcpy(copy.data, im.data, sizeof(im.data));
    int i;
    for (i = 0; i < (im.h * im.w * im.c); i++) {
	copy.data[i] = im.data[i];
    }
    return copy;
}

image rgb_to_grayscale(image im)
{
    // y = 0.299 R + 0.587 G + .114 B
    assert(im.c == 3);
    int x,y;
    float r,g,b;
    image gray = make_image(im.w, im.h, 1);
    for (x=0; x<im.w; x++) {
    for (y=0; y<im.h; y++) {
        r = im.data[x + y*im.w + 0*im.w*im.h];
        g = im.data[x + y*im.w + 1*im.w*im.h];
        b = im.data[x + y*im.w + 2*im.w*im.h];
        gray.data[x + y*im.w] = 0.299*r + 0.587*g + 0.114*b;
    }
    }
    return gray;
}

void shift_image(image im, int c, float v)
{
    // change the input image in-place (do not create a separate image)
    // wherever the return type of function is void
    int x,y,pos;
    for (x=0; x<im.w; x++) {
    for (y=0; y<im.h; y++) {
        pos = x + y*im.w + c*im.w*im.h;
        im.data[pos] = im.data[pos] + v;
    }
    }
}

// c-basic-offset
void clamp_image(image im)
{
    int i;
    float v;
    int length;
    length = im.h * im.w * im.c;
    for (i = 0; i < length; i++) {
      v = im.data[i];
      if (v < 0) {
        im.data[i] = 0;
      }
      if (v > 1) {
        im.data[i] = 1;
      }
    }
}

// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im) {
    // use three_way_max() and three_way_min()
    // V = max(R,G,B)
    // m = min(R,G,B)
    // C = V - m
    // S = C / V
    // H' = undef, C = 0
    // H' = (G-B)/C, if V = R
    // H' = (B-R)/C + 2, if V = G
    // H' = (R-G)/C + 4, if V = B
    // H = H'/6 + 1, if H' < 0
    // H = H'/6, otherwise
    int x, y;
    float R, G, B;
    float m, C, Hprime;
    float H, S, V;
    for (x = 0; x < im.w; x++) {
    for (y = 0; y < im.h; y++) {
        R = im.data[x + y * im.w + 0 * im.w * im.h];
        G = im.data[x + y * im.w + 1 * im.w * im.h];
        B = im.data[x + y * im.w + 2 * im.w * im.h];
        V = three_way_max(R, G, B);
        if (V == 0) {
          S = 0;
        } else {
          m = three_way_min(R, G, B);
          C = V - m;
          S = C / V;
        }
        if (C == 0) {
          H = 0;
        } else {
          if (V == R) {
            Hprime = (G-B)/C;
          } else if (V == G) {
            Hprime = (B-R)/C + 2;
          } else if (V == B) {
            Hprime = (R-G)/C + 4;
          } else {
            printf("error\n");
            return;
          }
          if (Hprime < 0) {
            H = Hprime / 6 + 1;
          } else {
            H = Hprime / 6;
          }
        }
        im.data[x + y * im.w + 0 * im.w * im.h] = H;
        im.data[x + y * im.w + 1 * im.w * im.h] = S;
        im.data[x + y * im.w + 2 * im.w * im.h] = V;
    }
    }
}

void hsv_to_rgb(image im) {
    // convert HSV to RGB using the given table
    int x, y;
    int Hi;
    float R, G, B;
    float H, S, V;
    float P, Q, F, T;
    for (x = 0; x < im.w; x++) {
    for (y = 0; y < im.h; y++) {
        H = im.data[x + y * im.w + 0 * im.w * im.h];
        S = im.data[x + y * im.w + 1 * im.w * im.h];
        V = im.data[x + y * im.w + 2 * im.w * im.h];

        H = H * 6;
        Hi = floor(H);
        F = H - Hi;
        P = V * (1 - S);
        Q = V * (1 - F * S);
        T = V * (1 - (1 - F) * S);
        if (Hi == 0) {
          R = V; G = T; B = P;
        }
        if (Hi == 1) {
          R = Q; G = V; B = P;
        }
        if (Hi == 2) {
          R = P; G = V; B = T;
        }
        if (Hi == 3) {
          R = P; G = Q; B = V;
        }
        if (Hi == 4) {
          R = T; G = P; B = V;
        }
        if (Hi == 5) {
          R = V; G = P; B = Q;
        }

        im.data[x + y * im.w + 0 * im.w * im.h] = R;
        im.data[x + y * im.w + 1 * im.w * im.h] = G;
        im.data[x + y * im.w + 2 * im.w * im.h] = B;
    }
    }
}

void scale_image(image im, int c, float v) {
  // to scale a channel by a certain amount
  // multiply each pixel of im in channel c with value v
    for (int x=0; x<im.w; x++) {
    for (int y=0; y<im.h; y++) {
      int pos = x + y*im.w + c*im.w*im.h;
      im.data[pos] = im.data[pos]*v;
    }
    }
}

void rgb_to_xyz(image im) {
    assert(im.c >= 3);
    int x, y;
    float R, G, B;
    for (x = 0; x < im.w; x++) {
        for (y = 0; y < im.h; y++) {
            R = get_pixel(im, x, y, 0);
            G = get_pixel(im, x, y, 1);
            B = get_pixel(im, x, y, 2);

            R = R / 255.0;
            G = G / 255.0;
            B = B / 255.0;

            R = (R > 0.04045) ? pow((R + 0.055)/1.055, 2.4) : R / 12.92;
            G = (G > 0.04045) ? pow((G + 0.055)/1.055, 2.4) : G / 12.92;
            B = (B > 0.04045) ? pow((B + 0.055)/1.055, 2.4) : B / 12.92;

            float X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B;
            float Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B;
            float Z = 0.0193339 * R + 0.1191920 * G + 0.9503041 * B;

            set_pixel(im, x, y, 0, X * 100.0);
            set_pixel(im, x, y, 1, Y * 100.0);
            set_pixel(im, x, y, 2, Z * 100.0);
        }
    }
}

void xyz_to_lab(image im) {
    assert(im.c >= 3);
    int x, y;
    float X, Y, Z;
    float Xn = 95.047;  
    float Yn = 100.0;
    float Zn = 108.883;
    for (x = 0; x < im.w; x++) {
        for (y = 0; y < im.h; y++) {
            X = get_pixel(im, x, y, 0) / Xn;
            Y = get_pixel(im, x, y, 1) / Yn;
            Z = get_pixel(im, x, y, 2) / Zn;

            X = (X > 0.008856) ? pow(X, 1.0/3.0) : (7.787 * X) + (16.0/116.0);
            Y = (Y > 0.008856) ? pow(Y, 1.0/3.0) : (7.787 * Y) + (16.0/116.0);
            Z = (Z > 0.008856) ? pow(Z, 1.0/3.0) : (7.787 * Z) + (16.0/116.0);

            float L = (116 * Y) - 16;
            float a = 500 * (X - Y);
            float b = 200 * (Y - Z);
            set_pixel(im, x, y, 0, L);
            set_pixel(im, x, y, 1, a);
            set_pixel(im, x, y, 2, b);
        }
    }
}

void lab_to_lch(image im) {
    assert(im.c >= 3);
    int x, y;
    for (x = 0; x < im.w; x++) {
        for (y = 0; y < im.h; y++) {
            float L = get_pixel(im, x, y, 0);
            float a = get_pixel(im, x, y, 1);
            float b = get_pixel(im, x, y, 2);

            float C = sqrt(a * a + b * b);
            float H = atan2(b, a);
            H = H * 180.0 / M_PI; 
            if (H < 0) H += 360.0;

            set_pixel(im, x, y, 0, L);
            set_pixel(im, x, y, 1, C);
            set_pixel(im, x, y, 2, H);
        }
    }
}

void rgb_to_lch(image im)
{
    // Note, this will involve gamma decompression, converting to CIEXYZ, converting to CIELUV, converting to HCL, and the reverse transformations
    // The upside is a similar colorspace to HSV but with better perceptual properties!
    // convert R,G,B values of image pixelwise L,C,H using the formulas
    rgb_to_xyz(im);
    xyz_to_lab(im);
    lab_to_lch(im);
}
