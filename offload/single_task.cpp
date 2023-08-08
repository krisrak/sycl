// SYCL, offload computation to GPU

#include <sycl/sycl.hpp>

int main(){
  // create a sycl queue with GPU device
  sycl::queue q (sycl::gpu_selector_v);

  // initialize an array
  const int N = 1024;
  // float data[N];
  // allocate data that can be accessed by host and device
  auto data = sycl::malloc_shared<float>(N, q);
  for (int i=0; i<N; i++) data[i] = i;

  // offload compute on GPU using single_task
  q.single_task([=](){
    for (int i=0; i<N; i++) data[i] *= 5;
  }).wait();

  // print output
  for (int i=0; i<N; i++) std::cout << data[i] << " ";
  std::cout << "\n";

  sycl::free(data, q);
}
