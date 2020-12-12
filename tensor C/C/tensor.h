#pragma once

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

struct tensor_type{
  uint32_t dims;
  uint32_t* dim_sizes;
  uint32_t* shifts;
  float* data;
  uint32_t data_length;
};

typedef struct tensor_type tensor_t;

tensor_t* tensor_init(const uint32_t dims, const uint32_t* dim_sizes, const float* data){
  tensor_t* res = malloc(sizeof(tensor_t));
  res->data_length=1;
  res->dims = dims;
  res->dim_sizes = malloc(dims*sizeof(uint32_t));
  res->shifts = malloc(dims*sizeof(uint32_t));
  memcpy(res->dim_sizes, dim_sizes, dims*sizeof(uint32_t));
  for(int i=0; i<dims; ++i){
    res->shifts[i]=1;
    for(int i=0; i<dims; ++i)
      res->data_length*=res->dim_sizes[i];
    for(int j=0;j<i;++j)
      res->shifts[j]*=res->dim_sizes[i];
    }
  res->data = malloc(sizeof(float)*res->data_length);
  if(data)
    memcpy(res->data, data, res->data_length*sizeof(float));
  else
    memset(res->data, 0, res->data_length*sizeof(float));
  return res;
}

void tensor_remove(tensor_t* t){
  free(t->data);
  free(t->dim_sizes);
  free(t->shifts);
}

tensor_t* add(tensor_t* t1, tensor_t* t2){
  if(t1->dims != t2->dims){
    return NULL;
  }
  for(int i=0;i<t1->dims; ++i){
    if(t1->dim_sizes[i] != t2->dim_sizes[i])
      return NULL;
  }
  tensor_t* res = tensor_init(t1->dims, t1->dim_sizes, NULL);
  for(int i=0;i<t1->data_length;++i){
    res->data[i] = t1->data[i]+t2->data[i];
  }
  return res;
}

tensor_t* mul(tensor_t* t1, const float scalar){
  tensor_t* res = tensor_init(t1->dims, t1->dim_sizes, NULL);
  for(int i=0;i<t1->data_length;++i){
    res->data[i] = t1->data[i]*scalar;
  }
  return res;
}

float get_value(tensor_t* t1, const uint32_t* coords){
  uint32_t start_index = 0;
  for(int j = 0; j<t1->dims; ++j)
    start_index+=coords[j]*t1->shifts[j];
  return t1->data[start_index];
}

void set_value(tensor_t* t1, const float val, const uint32_t* coords){
  uint32_t start_index = 0;
  for(int j = 0; j<t1->dims; ++j)
    start_index+=coords[j]*t1->shifts[j];
  t1->data[start_index] = val;
}


void sub_change_coord(tensor_t* t1, tensor_t* nn, int index, uint32_t* state, const uint32_t* order){
  if(index == (t1->dims-1)){
    uint32_t start_index = 0;
    for(int j = 0; j<(t1->dims-1); ++j)
      start_index+=state[j]*t1->shifts[j];
    for(int i=0; i<t1->dim_sizes[index]; ++i){
      state[index] = i;
      uint32_t* new_state = malloc(sizeof(uint32_t)*nn->dims);
      for(int i=0;i<t1->dims;++i){
        new_state[i] = state[order[i]];
      }
      set_value(nn, get_value(t1, state), new_state);
      free(new_state);
    }
  }else{
    for(int i=0; i<t1->dim_sizes[index];++i){
      state[index] = i;
      sub_change_coord(t1, nn, index+1, state, order);
    }
  }
}


tensor_t* changeCoord(tensor_t* t1, const uint32_t* order){
  uint32_t* new_dims = malloc(sizeof(uint32_t)*t1->dims);
  for(int i=0;i<t1->dims;++i){
    new_dims[i] = t1->dim_sizes[order[i]];
  }
  tensor_t* res = tensor_init(t1->dims, new_dims, t1->data);
  uint32_t* state = calloc(sizeof(uint32_t), t1->dims);
  sub_change_coord(t1, res, 0, state, order);
  free(state);
  free(new_dims);
  return res;
}

tensor_t* take(tensor_t* t1, const uint32_t* indeces, const uint32_t indeces_count){
  tensor_t* res = tensor_init(1, &indeces_count, NULL);
  for(int i=0; i<indeces_count; ++i){
    res->data[i] = t1->data[indeces[i]];
  }
  return res;
}

tensor_t* resape(tensor_t* t1, const uint32_t ndims, const uint32_t* ndim_sizes){
  tensor_t* res = tensor_init(ndims, ndim_sizes, t1->data);
  return res;
}

void out(tensor_t* t1, int index, uint32_t* state){
  if(index == (t1->dims-1)){
    uint32_t start_index = 0;
    for(int j = 0; j<(t1->dims-1); ++j)
      start_index+=state[j]*t1->shifts[j];
    printf("[");
    for(int i=0; i<t1->dim_sizes[index]; ++i){
      printf("%f ", t1->data[start_index+i]);
    }
    printf("]\n");
  }else{
    printf("[\n");
    for(int i=0; i<t1->dim_sizes[index];++i){
      state[index] = i;
      out(t1, index+1, state);
    }
    printf("]\n");
  }
}

void tensor_print(tensor_t* t){
  uint32_t* state = calloc(sizeof(uint32_t), t->dims);
  out(t, 0, state);
  free(state);
}
