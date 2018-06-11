//OpenMP version.  Edit and submit only this file.
/* Enter your details below
 * Name : Hermmy Wang
 * UCLA ID: 704978214
 * Email id: hermmyw@hotmail.com
 * Input: Old file
 */

/* Testing speeds:
1. 13.40x
2. 18.32x
3. 11.13x
4. 12.62x
5. 12.30x
6. 12.03x
7. 11.97x
8. 11.02x
9. 11.05x
10.10.17x
*/

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

int OMP_xMax;
#define xMax OMP_xMax
int OMP_yMax;
#define yMax OMP_yMax
int OMP_zMax;
#define zMax OMP_zMax

#define T 16
#define N 10
#define B 8
int OMP_Index(int x, int y, int z)
{
	return ((z * yMax + y) * xMax + x);
}
#define Index(x, y, z) OMP_Index(x, y, z)

double OMP_SQR(double x)
{
	return pow(x, 2.0);
}
#define SQR(x) OMP_SQR(x)

double* OMP_conv;
double* OMP_g;

void OMP_Initialize(int xM, int yM, int zM)
{
	xMax = xM;
	yMax = yM;
	zMax = zM;
	assert(OMP_conv = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
	assert(OMP_g = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
}
void OMP_Finish()
{
	free(OMP_conv);
	free(OMP_g);
}
void OMP_GaussianBlur(double *u, double Ksigma, int stepCount)
{
	double lambda = (Ksigma * Ksigma) / (double)(2 * stepCount);
	//double nu = (1.0 + 2.0*lambda - sqrt(1.0 + 4.0*lambda))/(2.0*lambda);
	double nu = 1.0 - sqrt(1.0 / lambda);
	int x, y, z, index, step, bx, by, bz;
	double boundryScale = 1.0 / (1.0 - nu);
	double postScale = pow(nu / lambda, (double)(3 * stepCount));

	for(step = 0; step < stepCount; step++)
	{
		#pragma omp parallel num_threads(T)
		{
			#pragma omp for private(x, y, z, index)
			for(z = 0; z < zMax; z++)
	  		{
				for(x = 0; x < xMax; x++)
			  	{
					for(y = 1; y < yMax; y++)
					{
						index = Index(x, y, z);
						u[index] += u[index - xMax] * nu;
					}
					for(y = yMax-2; y >= 0; y--)
					{
						index = Index(x, y, z);
						u[index] += u[index + xMax] * nu;
					}
				}
			}

			#pragma omp for private(x, y, z, index) nowait
			for (x = 0; x < xMax; x++) 
			{
				for(y = 0; y < yMax; y++)
				{
					for(z = 1; z < zMax; z++)
					{
					  index = Index(x, y, z);
					  u[index] = u[index - xMax*yMax] * nu;
					}
				}
			}
		}
	}



	#pragma omp paralel for num_threads(T) private (x, y, z) nowait
	for(z = 1; z < zMax; z+=B)
	{
		for(y = 0; y < yMax; y+=B)
		{
			for(x = 1; x < xMax; x+=B)
			{
				for (bz = z; (bz < z+B) && (bz < zMax); bz++) 
				{
					for (by = y; (by < y+B) && (by < yMax); by++) 
					{
						for (bx = x; (bx < x+B) && (bx < xMax); bx++) 
						{
							u[Index(bx, by, bz)] *= postScale;
						}
					}
				}
			}
		}
	}
}


void OMP_Deblur(double* u, const double* f, int maxIterations, double dt, double gamma, double sigma, double Ksigma)
{
	double epsilon = 1.0e-7;
	double sigma2 = SQR(sigma);
	int x, y, z, iteration, index;
	int converged = 0;
	int lastConverged = 0;
	int fullyConverged = (xMax - 1) * (yMax - 1) * (zMax - 1);
	double* conv = OMP_conv;
	double* g = OMP_g;
	double r, tmp;
	int bx, by, bz;
	for(iteration = 0; iteration < maxIterations && converged != fullyConverged; iteration++)
	{
		#pragma omp parallel for num_threads(N) private(x, y, z, bx, by, bz, index, tmp)
		for(z = 1; z < zMax - 1; z+=B)
		{
			for(y = 1; y < yMax - 1; y+=B)
			{
				for(x = 1; x < xMax - 1; x+=B)
				{
					for (bz = z; (bz < z+B) && (bz < zMax-1); bz++)
					{
						for (by = y; (by < y+B) && (by < yMax-1); by++)
						{
							for (bx = x; (bx < x+B) && (bx < xMax-1); bx++)
							{
								index = Index(bx, by, bz);
								tmp = u[index];
								g[index] = 1.0 / sqrt(epsilon +
										  SQR(tmp - u[index+1]) +
										  SQR(tmp - u[index-1]) +
										  SQR(tmp - u[index+xMax]) +
										  SQR(tmp - u[index-xMax]) +
										  SQR(tmp - u[index+xMax*yMax]) +
										  SQR(tmp - u[index-xMax*yMax]));
							}
						}
					}
				}
			}
		}

		memcpy(conv, u, sizeof(double) * xMax * yMax * zMax);
		OMP_GaussianBlur(conv, Ksigma, 3);

		#pragma omp parallel for num_threads(N) private(x, y, z, r, index)
		for(z = 1; z < zMax; z++)
		{
			for(y = 0; y < yMax; y++)
			{
				for(x = 1; x < xMax; x++)
				{
					index = Index(x, y, z);
					r = conv[index] * f[index] / sigma2;
					r = (r * (2.38944 + r * (0.950037 + r))) / (4.65314 + r * (2.57541 + r * (1.48937 + r)));
					conv[index] -= f[index] * r;
				}
			}
		}
		OMP_GaussianBlur(conv, Ksigma, 3);
		converged = 0;




	  
		int center, front, back, left, right, up, down;
		for(z = 1; z < zMax - 1; z++)
		{
			for(y = 1; y < yMax - 1; y++)
			{
				for(x = 1; x < xMax - 1; x++)
				{
					center = Index(x,y,z);
					front = center - 1;
					back = center + 1;
					left = center - xMax;
					right = center + xMax;
					up = center - xMax*yMax;
					down = center + xMax*yMax;
					double oldVal = u[center];
					double newVal = (oldVal + dt * (
										u[front] * g[front] +
										u[back] * g[back] +
										u[left] * g[left] +
										u[right] * g[right] +
										u[up] * g[up] +
										u[down] * g[down] - gamma * conv[center])) /
					(1.0 + dt * (g[back] + g[front] + g[right] + g[left] + g[down] + g[up]));

					if(fabs(oldVal - newVal) < epsilon)
					{
						converged++;
					}

					u[center] = newVal;
				}
			}
		}
		if(converged > lastConverged)
		{
			printf("%d pixels have converged on iteration %d\n", converged, iteration);
			lastConverged = converged;
		}
	}

}

