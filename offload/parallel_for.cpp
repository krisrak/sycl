// SYCL, offload parallel computation to GPU

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

  // offload parallel compute on GPU using parallel_for
  q.parallel_for(N, [=](auto i){
    data[i] *= 5;
  }).wait();

  // print output
  for (int i=0; i<N; i++) std::cout << data[i] << " ";
  std::cout << "\n";

  sycl::free(data, q);
}
