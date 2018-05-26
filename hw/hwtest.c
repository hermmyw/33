#include <stdio.h>
#define D 1020
void transpose(int* dst, int* src, int dim) {
  int i, j;
  for(i = 0; i < dim; i++)
    for (j = 0; j < dim;j++){
      dst[j*dim + i] = src[i*dim + j];
    }
}

int main(){
  int a[D*D];
  int b[D*D];
  int i, j;
  for (i = 0; i < D; i++){
    for (j = 0; j < D; j++){
      a[i*D+j] = i*D+j + 100;
      b[i*D+j] = 0;
      //      printf("a[%d]: %d   ", i*D+j, a[i*D+j]);                                                                                                                                    
    }
    //printf("\n");                                                                                                                                                                       
  }
  transpose(b, a, D);

}
