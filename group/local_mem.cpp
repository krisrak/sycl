// SYCL, work_group local memory access

#include <sycl/sycl.hpp>

int main(){
  // create a sycl queue with GPU device
  sycl::queue q (sycl::gpu_selector_v);
  std::cout << "Offload Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "local_mem_size: " << q.get_device().get_info<sycl::info::device::local_mem_size>() << "\n";
  
  auto N = 1024; // global size
  auto B = 128; //work-group size

  int data[N];
  for (int i=0; i<N; i++) data[i] = i;

  sycl::buffer a_buf(data, sycl::range<1>(N));

  // offload parallel compute on GPU using parallel_for
  q.submit([&](sycl::handler &h){
    sycl::accessor A_global(a_buf, h);
    sycl::local_accessor<int> A_local(B, h);

    h.parallel_for(sycl::nd_range<1>{N, B}, [=](sycl::nd_item<1> item){
      auto i = item.get_global_id(0);
      auto x = item.get_local_id(0);
      
      // copy from global to local memory and add a barrier
      A_local[x] = A_global[i];
      sycl::group_barrier(item.get_group());
      
      // some computation
      int temp = 0;
      for (int k = 0; k < B; k++) {
        temp += A_local[k];
	//temp += A_global[(item.get_group_linear_id()*B) + k];
      }
      A_global[i] = temp;
    });
  });

  auto ha = sycl::host_accessor(a_buf, sycl::read_only);

  // print output
  for (int i=0; i<N; i++) std::cout << data[i] << " ";
  std::cout << "\n";

}
