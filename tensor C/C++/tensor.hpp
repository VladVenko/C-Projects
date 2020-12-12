#pragma once

#include <stdint.h>
#include <string.h>
#include <stdexcept>
#include <iostream>

class tensor_t{
public:
  tensor_t(const uint32_t dims, const uint32_t* dim_sizes, const float* data = NULL){
    this->dims = dims;
    this->dim_sizes = new uint32_t[dims];
    this->shifts = new uint32_t[dims];
    memcpy(this->dim_sizes, dim_sizes, dims*sizeof(uint32_t));
    for(int i=0; i<dims; ++i){
      shifts[i]=1;
    for(int i=0; i<dims; ++i)
      data_length*=this->dim_sizes[i];
    for(int j=0;j<i;++j)
        shifts[j]*=this->dim_sizes[i];
    }
    this->data = new float[data_length];
    if(data)
      memcpy(this->data, data, data_length*sizeof(float));
    else
      memset(this->data, 0, data_length*sizeof(float));
  }

  virtual ~tensor_t(){
    delete[] data;
    delete[] dim_sizes;
    delete[] shifts;
  }

  tensor_t operator+(const tensor_t& t){
    if(dims != t.dims){
      throw std::runtime_error("can`t add tensors with different dimensions");
    }
    for(int i=0;i<dims; ++i){
      if(dim_sizes[i] != t.dim_sizes[i]){
        throw std::runtime_error("can`t add tensors with different dimensions");
      }
    }
    tensor_t res(dims, dim_sizes);
    for(int i=0;i<data_length;++i){
      res.data[i] = data[i]+t.data[i];
    }
    return res;
  }

  tensor_t operator*(const float& scalar){
    tensor_t res(dims, dim_sizes);
    for(int i=0;i<data_length;++i){
      res.data[i] = data[i]*scalar;
    }
    return res;
  }

  tensor_t changeCoord(const uint32_t* order){
    uint32_t* new_dims = new uint32_t[dims];
    for(int i=0;i<dims;++i){
      new_dims[i] = dim_sizes[order[i]];
    }
    tensor_t res(dims, new_dims, data);
    uint32_t* state = new uint32_t[dims];
    for(int i=0;i<dims;++i){
      state[i]=0;
    }
    sub_change_coord(res, 0, state, order);
    delete[] state;
    delete[] new_dims;
    return res;
  }

  tensor_t take(const uint32_t* indeces, const uint32_t indeces_count){
    tensor_t res(1, &indeces_count);
    for(int i=0; i<indeces_count; ++i){
      res.data[i] = data[indeces[i]];
    }
    return res;
  }

  tensor_t resape(const uint32_t ndims, const uint32_t* ndim_sizes){
    tensor_t res(ndims, ndim_sizes, data);
    return res;
  }

  friend tensor_t operator*(const float& scalar, const tensor_t& t);
  friend std::ostream& operator<<(std::ostream& s, const tensor_t& t);
private:
  uint32_t dims;
  uint32_t* dim_sizes;
  uint32_t* shifts;
  float* data;
  uint32_t data_length=1;

  void sub_change_coord(tensor_t& nn, int index, uint32_t* state, const uint32_t* order){
    if(index == (dims-1)){
      uint32_t start_index = 0;
      for(int j = 0; j<(dims-1); ++j)
        start_index+=state[j]*shifts[j];
      for(int i=0; i<dim_sizes[index]; ++i){
        state[index] = i;
        uint32_t* new_state = new uint32_t[nn.dims];
        for(int i=0;i<dims;++i){
          new_state[i] = state[order[i]];
        }
        nn.set_value(get_value(state), new_state);
        delete[] new_state;
      }
    }else{
      for(int i=0; i<dim_sizes[index];++i){
        state[index] = i;
        sub_change_coord(nn, index+1, state, order);
      }
    }

  }

  void out(std::ostream& s, int index, uint32_t* state) const{
    std::string indent = "";
    for(int i=0;i<index;++i){
      indent+="    ";
    }
    if(index == (dims-1)){
      uint32_t start_index = 0;
      for(int j = 0; j<(dims-1); ++j)
        start_index+=state[j]*shifts[j];
      s << indent << "[";
      for(int i=0; i<dim_sizes[index]; ++i){
        s << data[start_index+i] << " ";
      }
      s << "]" << std::endl;
    }else{
      s << indent << "[" << std::endl;
      for(int i=0; i<dim_sizes[index];++i){
        state[index] = i;
        out(s, index+1, state);
      }
      s << indent << "]" << std::endl;
    }
  }

  float get_value(const uint32_t* coords){
    uint32_t start_index = 0;
    for(int j = 0; j<dims; ++j)
      start_index+=coords[j]*shifts[j];
    return data[start_index];
  }

  void set_value(const float val, const uint32_t* coords){
    uint32_t start_index = 0;
    for(int j = 0; j<dims; ++j)
      start_index+=coords[j]*shifts[j];
    data[start_index] = val;
  }
};

tensor_t operator*(const float& scalar, const tensor_t& t){
  tensor_t res(t.dims, t.dim_sizes);
  for(int i=0;i<t.data_length;++i){
	res.data[i] = t.data[i]*scalar;
  }
  return res;
}

std::ostream& operator<<(std::ostream& s, const tensor_t& t){
  uint32_t* state = new uint32_t[t.dims];
  for(int i=0;i<t.dims;++i){
    state[i]=0;
  }
  t.out(s, 0, state);
  delete[] state;
  return s;
}
