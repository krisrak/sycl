// SYCL, USM with in_order queue propery

#include <sycl/sycl.hpp>

int main(){
  // create a sycl queue with GPU device and in_order queue property
  sycl::queue q (sycl::gpu_selector_v, sycl::property::queue::in_order());
  std::cout << "Offload Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  // initialize an array
  const int N = 1024;
  float data[N];
  for (int i=0; i<N; i++) data[i] = i;

  // allocate data that can be accessed by device
  auto data_device = sycl::malloc_device<float>(N, q);

  // move memory from host to device
  q.memcpy(data_device, data, N*sizeof(float));
  
  // offload parallel compute on GPU using parallel_for
  q.parallel_for(N, [=](auto i){
    data_device[i] *= 5;
  });

  // move memory from device to host
  q.memcpy(data, data_device, N*sizeof(float));

  q.wait();

  // print output
  for (int i=0; i<N; i++) std::cout << data[i] << " ";
  std::cout << "\n";

  sycl::free(data_device, q);
}
