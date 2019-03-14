///////////////////////////////////////////////////////////////////////////////////
// File : APSF.cpp
///////////////////////////////////////////////////////////////////////////////////
//
// LumosQuad - A Lightning Generator
// Copyright 2007
// The University of North Carolina at Chapel Hill
// 
///////////////////////////////////////////////////////////////////////////////////
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
// 
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//  The University of North Carolina at Chapel Hill makes no representations 
//  about the suitability of this software for any purpose. It is provided 
//  "as is" without express or implied warranty.
//
//  Permission to use, copy, modify and distribute this software and its
//  documentation for educational, research and non-profit purposes, without
//  fee, and without a written agreement is hereby granted, provided that the
//  above copyright notice and the following three paragraphs appear in all
//  copies.
//
//  THE UNIVERSITY OF NORTH CAROLINA SPECIFICALLY DISCLAIM ANY WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
//  FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS ON AN
//  "AS IS" BASIS, AND THE UNIVERSITY OF NORTH CAROLINA HAS NO OBLIGATION TO
//  PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
//
//  Please send questions and comments about LumosQuad to kim@cs.unc.edu.

#include <cmath>
#include <cassert>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include "apsf.h"

using namespace std;

class APSF  
{
public:
	APSF(int res = 512);

  //! read in an APSF file
  void read(const char* filename);
  //! write out an APSF file
  void write(const char* filename);
  
  //! generate one line of the kernel and spin it radially
  void generateKernelFast();

  //! resolution of current kernel
  int res() { return _res; };
  
  //! returns the float array for the kernel
  float* kernel() { return _kernel; };
  
private:
  //! kernel resolution
  int     _res;
  //! convolution kernel
  float*  _kernel;

  ////////////////////////////////////////////////////////////////////
  // APSF components
  ////////////////////////////////////////////////////////////////////

  // scattering parameters
  float _q;
  float _T;
  float _I0;
  float _sigma;
  float _R;
  float _D;

  float _retinaSize;
  float _eyeSize;
  
  //! number of coefficients
  int _maxTerms;
  
  //! function value at a point
  float pointAPSF(float mu);
  
  ////////////////////////////////////////////////////////////////////
  // auxiliary functions
  ////////////////////////////////////////////////////////////////////
  float legendreM(int m, float mu);
  float gM(float I0, int m) {
    return (m == 0) ? 0.0f : exp(-(betaM(m, _q) * _T + alphaM(m) * log(_T)));
  };
  float alphaM(float m) { 
    return m + 1.0f;
  };
  float betaM(float m, float q) {
    return ((2.0f * m + 1.0f) / m) * (1.0f - pow(q, (int)m - 1));
  };
  float factorial(float x) {
    return (x <= 1.0f) ? 1.0f : x * factorial(x - 1.0f);
  };
  float choose(float x, float y) {
    return factorial(x) / (factorial(y) * factorial(x - y));
  };
};

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

APSF::APSF(int res) :
  _res(res)
{
  assert (res % 2 == 1);

  cudaMallocManaged(&_kernel, sizeof(float) * _res * _res);

  _q = 0.999;
  _R = 400.0f;
  _D = 2000.0f;
  _T = 1.001f;
  _sigma = 0.5f;
  
  _maxTerms = 600;
  _I0 = 1.0f;
  _retinaSize = 0.01f;
  _eyeSize = 0.025f;
}

//////////////////////////////////////////////////////////////////////
// Legendre polymonial
//////////////////////////////////////////////////////////////////////
float APSF::legendreM(int m, float mu)
{
  vector<float> memoized;
  memoized.push_back(1.0f);
  memoized.push_back(mu);
 
  for (int x = 2; x <= m; x++)
  {
    float newMemo = ((2.0f * (float)x - 1.0f) * mu * memoized[x - 1] - 
                             ((float)x - 1.0f) * memoized[x - 2]) / (float)x;
    memoized.push_back(newMemo);
  }

  return memoized[m];
}

//////////////////////////////////////////////////////////////////////
// scattering function at a point
//////////////////////////////////////////////////////////////////////
float APSF::pointAPSF(float mu)
{
  float total = 0.0f;

  for (int m = 0; m < _maxTerms; m++)
    total += (gM(_I0, m) + gM(_I0, m + 1)) * legendreM(m, mu);
    
  return total;
}

//////////////////////////////////////////////////////////////////////
// generate a convolution kernel
//////////////////////////////////////////////////////////////////////
void APSF::generateKernelFast()
{
  float dx = _retinaSize / (float)_res;
  float dy = _retinaSize / (float)_res;
  int halfRes = _res / 2;
  float* oneD = new float[_res];
  
  float max = 0.0f;
  float min = 1000.0f;
  int x,y = halfRes;
  for (x = 0; x < _res; x++)
  {
    // calc angle
    float diffX = (x - halfRes) * dx;
    float diffY = (y - halfRes) * dy;
    float distance = sqrt(diffX * diffX + diffY * diffY);
    if ((distance / _eyeSize) > (_R / _D))
      oneD[x] = 0.0f;
    else
    {
      float i = -distance * distance * _D * _D + _eyeSize * _eyeSize * _R * _R + distance * distance * _R * _R;
      i = _eyeSize * _eyeSize * _D - _eyeSize * sqrt(i);
      i /= _eyeSize * _eyeSize + distance * distance;
      float mu = M_PI - atan(_retinaSize / distance) - asin((_D - i) / _R);
      oneD[x] = pointAPSF(cos(mu));
    
      min = (oneD[x] < min) ? oneD[x] : min;
    }
    max = (oneD[x] > max) ? oneD[x] : max;
  }
  
  // floor 
  if (min > 0.0f)
  {
    for (int i = 0; i < _res; i++)
      if (oneD[i] > 0.0f)
        oneD[i] -= min;
    max -= min;
  }
  
  // normalize
  if (max > 1.0f)
  {
    float maxInv = 1.0f / max;
    for (int i = 0; i < _res; i++)
      oneD[i] *= maxInv;
  }

  // interpolate the kernel
  int index = 0;
  for (y = 0; y < _res; y++)
    for (x = 0; x < _res; x++, index++)
    {
      float dx = fabs((float)(x - halfRes));      
      float dy = fabs((float)(y - halfRes));
      float magnitude = sqrtf(dx * dx + dy * dy);
      
      int lower = floor(magnitude);
      if (lower > halfRes - 1)
      {
        _kernel[index] = 0.0f;
        continue;
      }
      float lerp = magnitude - lower;
      _kernel[index] = (1.0f - lerp) * oneD[halfRes + lower] +
                               lerp  * oneD[halfRes + lower + 1];
      
    }

  delete[] oneD;
}

//////////////////////////////////////////////////////////////////////
// save the kernel in binary
//////////////////////////////////////////////////////////////////////
void APSF::write(const char* filename)
{
  // open file
  FILE* file;
  file = fopen(filename, "wb");

  fwrite(_kernel,     sizeof(float) * _res * _res, 1, file);

  fclose(file);
}

float *make_apsf(int size)
{
  APSF kernel_gen(size);

  kernel_gen.generateKernelFast();

  return kernel_gen.kernel();
}
