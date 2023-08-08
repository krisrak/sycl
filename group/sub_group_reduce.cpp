// SYCL, group algorithm on sub-group

#include <sycl/sycl.hpp>

int main(){
  // create a sycl queue with GPU device
  sycl::queue q (sycl::gpu_selector_v);
  std::cout << "Offload Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  
  auto N = 256; // global size
  auto B = 64; // work-group size

  auto data = sycl::malloc_shared<int>(N, q);
  for(int i=0;i<N;i++) data[i] = i;

  // offload parallel compute on GPU using parallel_for
  q.parallel_for(sycl::nd_range<1>(N, B), [=](sycl::nd_item<1> item){
    auto i = item.get_global_id();
    auto sg = item.get_sub_group();

    data[i] = sycl::reduce_over_group(sg, data[i], sycl::plus<>());
  }).wait();

  for(int i=0;i<N;i++) std::cout << data[i] << " ";
  std::cout << "\n";
}
