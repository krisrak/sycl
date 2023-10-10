// SYCL, event based dependency between kernels

#include <sycl/sycl.hpp>

int main(){
  // create a sycl queue with GPU device
  sycl::queue q (sycl::gpu_selector_v);
  std::cout << "Offload Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  // initialize arrays
  const int N = 1024;
  float a[N], b[N], c[N];
  for (int i=0; i<N; i++) {
      a[i] = 2;
      b[i] = 3;
  }

  // allocate data that can be accessed by device
  auto a_device = sycl::malloc_device<float>(N, q);
  auto b_device = sycl::malloc_device<float>(N, q);
  auto c_device = sycl::malloc_device<float>(N, q);

  // move memory from host to device
  auto e1 = q.memcpy(a_device, a, N*sizeof(float));
  auto e2 = q.memcpy(b_device, b, N*sizeof(float));
  
  // offload parallel compute on GPU using parallel_for
  auto e3 = q.parallel_for(N, {e1, e2}, [=](auto i){
    c_device[i]= a_device[i] + b_device[i];
  });

  // move memory from device to host
  q.memcpy(c, c_device, N*sizeof(float), {e3}).wait();

  // print output
  for (int i=0; i<N; i++) std::cout << c[i] << " ";
  std::cout << "\n";

  sycl::free(a_device, q);
  sycl::free(b_device, q);
  sycl::free(c_device, q);
}
