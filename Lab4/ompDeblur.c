//OpenMP version.  Edit and submit only this file.
/* Enter your details below
 * Name : Hermmy Wang 
 * UCLA ID: 704978214
 * Email id: hermmyw@hotmail.com
 * Input: New Files
 */

/* Speedup Test:
lnxsrv07: (6/8 15:56)
14.806616x OpenMP took 0.901620 time units
13.154899x OpenMP took 0.876574 time units
11.876285x OpenMP took 0.977380 time units
11.991698x OpenMP took 0.957356 time units
12.082893x OpenMP took 0.931644 time units

lnxsrv09: (6/8 15:56)
16.310818x OpenMP took 0.964568 time units
12.688093x OpenMP took 0.945574 time units
13.555041x OpenMP took 0.912845 time units
14.742818x OpenMP took 0.951970 time units
14.043162x OpenMP took 0.958608 time units
==========

lnxsrv08:
10.495766x
10.784907x
9.036929x
9.136095x
9.324762x
9.275296x
8.891788x
10.726199x OpenMP took 0.898383 time units

*/
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
int OMP_xMax;
#define xMax OMP_xMax
int OMP_yMax;
#define yMax OMP_yMax
int OMP_zMax;
#define zMax OMP_zMax
#define T 16
#define N 16
#define B 8

double OMP_SQR(double x)
{
	return (x * x);
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
	double lambda = 0.54;
	double nu = (1.0 + lambda*2.0 - sqrt(1.0 + lambda*4.0))/(lambda*2.0);
	int x, y, z, step, index;
	double boundryScale = 1.0 / (1.0 - nu);
	double postScale = pow(nu / lambda, 9.0);
	int yxMax = (1<<14);
	int zyxMax = (1<<21) - yxMax;
	int zy, zx, yx;
	int edge;

	for(step = 0; step < stepCount; step++)
	{
		#pragma omp parallel num_threads(T)
		{
			#pragma omp for private(y,x,index,zy)
			for(z = 0; z < zMax; z++)
			{
				zy = z<<14;
				for(y = 0; y < yMax; y++)
				{
					u[zy] *= boundryScale;
					index = zy;
					for(x = 1; x < xMax; x++)
					{
						index++;
						u[index] += u[index-1] * nu;
					}
					u[zy] *= boundryScale;
					index = zy + xMax - 1;
					for(x = xMax - 2; x >= 0; x--)
					{
						index--;
						u[index] += u[index+1] * nu;
					}
					zy+=xMax;
				}
			}

			edge = yxMax-xMax;
			#pragma omp for private(z,x,y,index,zx)
			for(z = 0; z < zMax; z++)
			{
				zx = z<<14;
				for (x = 0; x < xMax; x++)
				{
					u[zx] *= boundryScale;
					index = zx;
					for(y = 1; y < yMax; y++)
					{
						index+=xMax;
						u[index] += u[index-xMax] * nu;
					}
					u[zx+edge] *= boundryScale;
					index = zx + yxMax - xMax;
					for(y = yMax - 2; y >= 0; y--)
					{
						index -= xMax;
						u[index] += u[index+xMax] * nu;
					}
					zx++;
				}
			}

			edge = zyxMax;
			#pragma omp for private(y,x,z,index,yx)
			for(y = 0; y < yMax; y++)
			{
				/* (z*xMax*yMax + y*xMax + x) */
				yx = y<<7;
				for (x = 0; x < xMax; x++)
				{
					u[yx] *= boundryScale;
					index = yx;
					for(z = 1; z < zMax; z++)
					{
						index+=yxMax;;
						u[index] = u[index-yxMax] * nu;
					}
					u[yx+edge] *= boundryScale;
					index = yx+zyxMax;
					for(z = zMax - 2; z >= 0; z--)
					{
						index-=yxMax;
						u[index] += u[index+yxMax] * nu;
					}
					yx++;
				}
			}
		}
	}

	#pragma omp parallel for num_threads(T) private(x,y,z,zy,index)
	for(z = 0; z < zMax; z++)
	{
		zy = z<<14;
		for (y = 0; y < yMax; y++)
		{
			index=zy;
			for(x = 0; x < xMax; x+=4)
			{
				u[index++] *= postScale;
				u[index++] *= postScale;
				u[index++] *= postScale;
				u[index++] *= postScale;
			}
			zy+=xMax;
		}
	}

}
void OMP_Deblur(double* u, const double* f, int maxIterations, double dt, double gamma, double sigma, double Ksigma)
{
	double epsilon = 1.0e-7;
	double sigma2 = SQR(sigma);
	int x, y, z, iteration, index, zy, by, bx;
	int converged = 0;
	int lastConverged = 0;
	int fullyConverged = (xMax - 1) * (yMax - 1) * (zMax - 1);
	double* conv = OMP_conv;
	double* g = OMP_g;
	int yxMax = 1<<14;

	for(iteration = 0; iteration < maxIterations && converged != fullyConverged; iteration++)
	{
		#pragma omp parallel for num_threads(N) private(z, y, x, index)
		for(z = 1; z < zMax - 1; z++)
		{
			int zy = z<<14;
			for (y = 1; y < yMax - 1; y++)
			{
				index=zy+=xMax;
				for(x = 1; x < xMax - 1; x+=2)
				{
					index++;
					double tmp1 = u[index];
					double tmp2 = u[index+1];
					double tmp3 = SQR(tmp1 - tmp2);
					g[index] = 1.0 / sqrt(epsilon + 
						tmp3 + 
						SQR(tmp1 - u[index-1]) + 
						SQR(tmp1 - u[index+xMax]) + 
						SQR(tmp1 - u[index-xMax]) + 
						SQR(tmp1 - u[index+yxMax]) + 
						SQR(tmp1 - u[index-yxMax]));

					index++;
					g[index] = 1.0 / sqrt(epsilon + 
						SQR(tmp2 - u[index+1]) + 
						tmp3 + 
						SQR(tmp2 - u[index+xMax]) + 
						SQR(tmp2 - u[index-xMax]) + 
						SQR(tmp2 - u[index+yxMax]) + 
						SQR(tmp2 - u[index-yxMax]));
				}
			}
		}


		memcpy(conv, u, (1<<24));
		OMP_GaussianBlur(conv, Ksigma, 3);

		#pragma omp parallel for num_threads(N) private(y, x, index, bx, by)
		for(y = 0; y < yMax; y+=B)
		{
			for(x = 0; x < xMax; x+=B)
			{
				int yx = y<<7;
				for (by = y; (by < y+B && by < yMax); by++)
				{
					index = yx+x;
					for (bx = x; (bx < x+B && bx < xMax); bx++)
					{
						double c = conv[index];
						double fi = f[index];
						double r = c * fi / sigma2;
						r = (r * (2.38944 + r * (0.950037 + r))) / (4.65314 + r * (2.57541 + r * (1.48937 + r)));
						c -= fi * r;
						conv[index] = c;
						index++;
					}
					yx+=xMax;
				}
			}
		}



		OMP_GaussianBlur(conv, Ksigma, 3);
		converged = 0;
		
		int z3=yxMax;
		for(z = 1; z < zMax - 1; z++)
		{
			int y3=xMax;
			for(y = 1; y < yMax - 1; y++)
			{
				zy = z3+y3+1;
				for(x = 1; x < xMax - 1; x+=2)
				{
					int center = zy;
					int front = center - 1;
					int back = center + 1;
					int left = center - xMax;
					int right = center + xMax;
					int up = center - yxMax;
					int down = center + yxMax;
					double oldVal = u[center];
					double nextVal = u[back];

					double newVal = (oldVal + dt * ( 
						u[front] * g[front] + 
						nextVal * g[back] + 
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
					front = center;
					center = back;
					back = center + 1;
					left = center - xMax;
					right = center + xMax;
					up = center - yxMax;
					down = center + yxMax;

					//oldVal = nextVal;
					newVal = (nextVal + dt * ( 
						u[front] * g[front] + 
						u[back] * g[back] + 
						u[left] * g[left] + 
						u[right] * g[right] + 
						u[up] * g[up] + 
						u[down] * g[down] - gamma * conv[center])) /
						(1.0 + dt * (g[back] + g[front] + g[right] + g[left] + g[down] + g[up]));
					if(fabs(nextVal - newVal) < epsilon)
					{
						converged++;
					}
					u[center] = newVal;
					zy+=2;
				}
				y3+=xMax;
			}
			z3+=yxMax;
		}

		if(converged > lastConverged)
		{
			printf("%d pixels have converged on iteration %d\n", converged, iteration);
			lastConverged = converged;
		}
	}
}

