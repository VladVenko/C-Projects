#include "tensor.hpp"
#include <iostream>

int main(){
  uint32_t dims[] = {2, 2};
  float data1[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float data2[] = {2, 2, 2, 2, 2, 2, 2, 2};
  tensor_t a1(2, dims, data1);
  tensor_t a2(2, dims, data2);
  std::cout << a1 << std::endl;
  std::cout << a1+a2 << std::endl;
  std::cout << a1*2 << std::endl;
  std::cout << 5*a1 << std::endl;
  uint32_t new_ord[] = {1, 0};
  std::cout << a1.changeCoord(new_ord) << std::endl;
}
