// SYCL, buffer accessor memory model

#include <sycl/sycl.hpp>

int main(){
  // create a sycl queue with GPU device
  sycl::queue q (sycl::gpu_selector_v);
  std::cout << "Offload Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  // initialize an array
  const int N = 1024;
  float data[N];
  for (int i=0; i<N; i++) data[i] = i;

  // create buffer for data
  sycl::buffer buf_data(data, sycl::range(N));

  // offload parallel compute on GPU using accessor
  q.submit([&](sycl::handler &h){
    sycl::accessor acc_data(buf_data,h, sycl::read_write);
    h.parallel_for(N, [=](auto i){
      acc_data[i] *= 5;
    });
  });

  // host_accessor to copy back data to host
  sycl::host_accessor ha(buf_data, sycl::read_only);

  // print output
  for (int i=0; i<N; i++) std::cout << data[i] << " ";
  std::cout << "\n";
}
