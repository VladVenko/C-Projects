#include "tensor.h"


int main(){
  uint32_t dims[] = {2, 2};
  float data1[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float data2[] = {2, 2, 2, 2, 2, 2, 2, 2};
  tensor_t* a1 = tensor_init(2, dims, data1);
  tensor_t* a2 = tensor_init(2, dims, data2);
  tensor_print(a1);
  printf("\n");
  tensor_print(add(a1, a2));
  printf("\n");
  tensor_print(mul(a1, 2));
  printf("\n");
  uint32_t new_ord[] = {1, 0};
  tensor_print(changeCoord(a1, new_ord));
  printf("\n");
}
