#pragma once

#include <memory>
#include <sstream>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include "common/exception.h"
#include "common/base_matrix.h"
#include "gpu/types-gpu.h"
#include "handles.h"

namespace amunmt {
namespace GPU {
namespace mblas {

using namespace thrust::placeholders;

///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void gSum(const T *data, size_t count, T &ret)
{
  ret = 0;
  for (size_t i = 0; i < count; ++i) {
    ret += data[i];
  }
}

template<typename T>
T Sum(const T *data, size_t count)
{
  T ret;
  T *d_ret;
  HANDLE_ERROR( cudaMalloc(&d_ret, sizeof(T)) );

  const cudaStream_t& stream = CudaStreamHandler::GetStream();
  HANDLE_ERROR( cudaStreamSynchronize(stream));

  gSum<<<1, 1, 0, stream>>>(data, count, *d_ret);
  HANDLE_ERROR( cudaMemcpy(&ret, d_ret, sizeof(T), cudaMemcpyDeviceToHost) );
  HANDLE_ERROR(cudaFree(d_ret));

  HANDLE_ERROR( cudaStreamSynchronize(stream));
  HANDLE_ERROR( cudaDeviceSynchronize() );

  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class TMatrix : public BaseMatrix {
  public:
    typedef T value_type;

    TMatrix()
    : maxSize_(0)
    , data_(nullptr)
    {
      dim_[0] = 0;
      dim_[1] = 0;
      dim_[2] = 0;
      dim_[3] = 0;
      updateStrides();
    }

    TMatrix(size_t rows, size_t cols, size_t beam, size_t batches, bool zero = false)
    {
      dim_[0] = rows;
      dim_[1] = cols;
      dim_[2] = beam;
      dim_[3] = batches;
      updateStrides();
      maxSize_ = size_;

      HANDLE_ERROR( cudaMalloc(&data_, maxSize_ * sizeof(T)) );
      //std::cerr << "malloc data1:" << data_ << std::endl;
      if (zero) {
        HANDLE_ERROR( cudaMemsetAsync(data_, 0, maxSize_ * sizeof(T), CudaStreamHandler::GetStream()) );
      }
    }

    TMatrix(TMatrix&& m)
    : TMatrix()
    {
      swap(m);
    }

    TMatrix(const TMatrix& m)
    : maxSize_(m.maxSize_)
    , size_(m.size_)
    {
      dim_[0] = m.dim_[0];
      dim_[1] = m.dim_[1];
      dim_[2] = m.dim_[2];
      dim_[3] = m.dim_[3];

      stride_[0] = m.stride_[0];
      stride_[1] = m.stride_[1];
      stride_[2] = m.stride_[2];
      stride_[3] = m.stride_[3];

      HANDLE_ERROR( cudaMalloc(&data_, maxSize_ * sizeof(T)) );
      //std::cerr << "malloc data2:" << data_ << std::endl;
      HANDLE_ERROR( cudaMemcpyAsync(
          data_,
          m.data_,
          maxSize_ * sizeof(T),
          cudaMemcpyDeviceToDevice,
          CudaStreamHandler::GetStream()) );
    }

    ~TMatrix()
    {
      HANDLE_ERROR(cudaFree(data_));
    }

    virtual size_t dim(size_t i) const
    {
      return dim_[i];
    }

    void Resize(size_t rows, size_t cols, size_t beam = 1, size_t batches = 1) {
      size_t newSize = cols * rows * beam * batches;
      if (data_) {
        if (newSize > maxSize_) {
          T *newData;
          HANDLE_ERROR( cudaMalloc(&newData, newSize * sizeof(T)) );
          //std::cerr << "malloc data3:" << data_ << std::endl;

          //size_t count = std::min(maxSize_, newSize);

          HANDLE_ERROR( cudaMemcpyAsync(
              newData,
              data_,
              size() * sizeof(T),
              cudaMemcpyDeviceToDevice,
              CudaStreamHandler::GetStream()) );

          //std::cerr << "free data1:" << data_ << std::endl;
          HANDLE_ERROR(cudaFree(data_));
          data_ = newData;
          maxSize_ = newSize;
        }
        else if (rows == 0 || cols == 0) {
            HANDLE_ERROR(cudaFree(data_));
            data_ = nullptr;
            dim_[0] = 0;
            dim_[1] = 0;
            dim_[2] = 0;
            dim_[3] = 0;
            maxSize_ = 0;
        }
      }
      else {
        HANDLE_ERROR( cudaMalloc(&data_, newSize * sizeof(T)) );
        //std::cerr << "malloc data4:" << data_ << std::endl;
        maxSize_ = newSize;
      }

      dim_[0] = rows;
      dim_[1] = cols;
      dim_[2] = beam;
      dim_[3] = batches;

      updateStrides();
    }

    void NewSize(size_t rows, size_t cols, size_t beam = 1, size_t batches = 1) {
      size_t newSize = cols * rows * beam * batches;
      if (data_) {
        if (newSize > maxSize_) {
          T *newData;
          HANDLE_ERROR( cudaMalloc(&newData, newSize * sizeof(T)) );
          HANDLE_ERROR( cudaFree(data_));
          data_ = newData;
          maxSize_ = newSize;
        }
        else if (rows == 0 || cols == 0) {
            HANDLE_ERROR( cudaFree(data_));
            data_ = nullptr;
            dim_[0] = 0;
            dim_[1] = 0;
            dim_[2] = 0;
            dim_[3] = 0;
            maxSize_ = 0;
        }
      }
      else {
        HANDLE_ERROR( cudaMalloc(&data_, newSize * sizeof(T)) );
        //std::cerr << "malloc data4:" << data_ << std::endl;
        maxSize_ = newSize;
      }

      dim_[0] = rows;
      dim_[1] = cols;
      dim_[2] = beam;
      dim_[3] = batches;

      updateStrides();
    }

    void reserve(size_t size)
    {
      assert(data_ == nullptr);
      HANDLE_ERROR( cudaMalloc(&data_, size * sizeof(T)) );
      maxSize_ = size;
    }

    /*
    void ReduceDimensions()
    {
    	if (dim_[2] == 1) {
    		dim_[2] = dim_[3];
    		dim_[3] = 1;
    	}
    	if (dim_[0] == 1) {
    		dim_[0] = dim_[2];
    		dim_[2] = dim_[3];
    		dim_[3] = 1;
    	}
    	if (dim_[1] == 1) {
    		dim_[1] = dim_[0];
    		dim_[0] = dim_[2];
    		dim_[2] = dim_[3];
    		dim_[3] = 1;
    	}
    }
    */

    virtual std::string Debug(size_t verbosity = 1) const
    {
      std::stringstream strm;
      strm << BaseMatrix::Debug(verbosity) << " ";
      strm << data_ << " "
          << maxSize_ << " "
          << std::flush;

      if (verbosity) {
        T sum = Sum(data(), size());
        strm << "sum=" << sum << std::flush;

        if (verbosity == 2) {
          const cudaStream_t& stream = CudaStreamHandler::GetStream();
          T h_data[size()];

          HANDLE_ERROR( cudaMemcpyAsync(
              &h_data,
              data_,
              size() * sizeof(T),
              cudaMemcpyDeviceToHost,
              stream) );
          HANDLE_ERROR( cudaStreamSynchronize(stream) );

          for (size_t i = 0; i < size(); ++i) {
            strm << " " << h_data[i];
          }
        }
      }

      return strm.str();
    }

    size_t size() const
    { return size_; }

    value_type* data() {
      return data_;
    }

    const value_type* data() const {
      return data_;
    }

    void swap(TMatrix &other)
    {
      std::swap(dim_, other.dim_);
      std::swap(size_, other.size_);
      std::swap(maxSize_, other.maxSize_);
      std::swap(data_, other.data_);
    }

    uint stride(uint i) const
    {
      return stride_[i];
    }

    void updateStrides()
    {
      stride_[0] = dim_[1];
      stride_[1] = 1;
      stride_[2] = dim_[0] * dim_[1];
      stride_[3] = dim_[0] * dim_[1] * dim_[2];

      size_ = stride_[3] * dim_[3];
    }

    void id2Indices(uint id, uint *out) const
    {
      assert(id < size());

      out[3] = id / stride(3);
      id = id % stride(3);

      out[2] = id / stride(2);
      id = id % stride(2);

      out[0] = id / stride(0);
      id = id % stride(0);

      out[1] = id / stride(1);
    }

    uint indices2Id(uint a, uint b, uint c, uint d) const
    {
      assert(a < dim(0));
      assert(b < dim(1));
      assert(c < dim(2));
      assert(d < dim(3));

      uint ind = 0;
      ind += a * stride(0);
      ind += b * stride(1);
      ind += c * stride(2);
      ind += d * stride(3);

      assert(ind < size());
      return ind;
    }

    T get(uint a, uint b, uint c, uint d) const
    {
      uint id = indices2Id(a, b, c, d);

      T ret;
      HANDLE_ERROR( cudaMemcpy(&ret, data_ + id, sizeof(T), cudaMemcpyDeviceToHost) );
      //HANDLE_ERROR( cudaStreamSynchronize(CudaStreamHandler::GetStream()));

      return ret;
    }

    void set(const T &val, uint a, uint b, uint c, uint d)
    {
      uint id = indices2Id(a, b, c, d);

      HANDLE_ERROR( cudaMemcpy(data_ + id, &val, sizeof(T), cudaMemcpyHostToDevice) );
    }

  private:
    size_t dim_[SHAPE_SIZE];
    size_t stride_[SHAPE_SIZE];

    size_t size_;
    size_t maxSize_;
    T *data_;
};

typedef TMatrix<float> Matrix;
typedef TMatrix<uint> IMatrix;


}  // namespace mblas
}  // namespace GPU
}
