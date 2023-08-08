// SYCL, nd_range kernel with work-group size

#include <sycl/sycl.hpp>

int main(){
  // create a sycl queue with GPU device
  sycl::queue q (sycl::gpu_selector_v);
  std::cout << "Offload Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  auto N = 1024; // global size
  auto B = 128; //work-group size

  // allocate data that can be accessed by host and device
  auto data = sycl::malloc_shared<float>(N, q);
  for (int i=0; i<N; i++) data[i] = i;

  // offload parallel compute on GPU using parallel_for
  q.parallel_for(sycl::nd_range<1>(N, B), [=](sycl::nd_item<1> item){
    auto i = item.get_global_id();
    data[i] *= 5;
  }).wait();

  // print output
  for (int i=0; i<N; i++) std::cout << data[i] << " ";
  std::cout << "\n";

  sycl::free(data, q);
}
