// SYCL, print work-group info and indeses when 3d data

#include <sycl/sycl.hpp>

int main(){
  // create a sycl queue with GPU device
  sycl::queue q (sycl::gpu_selector_v);
  std::cout << "Offload Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  
  // print max supported work-group size
  std::cout << "Max Work-Group Size: " << q.get_device().get_info<sycl::info::device::max_work_group_size>() << "\n";

  auto N = sycl::range<3> {4, 4, 4}; // global size
  auto B = sycl::range<3> {2, 2, 2}; // work-group size

  // offload parallel compute on GPU using parallel_for
  q.submit([&](sycl::handler &h){
    // setup sycl stream class to print standard output from device code
    auto out = sycl::stream(4096, 4096, h);
    
    h.parallel_for(sycl::nd_range<3>(N, B), [=](sycl::nd_item<3> item){
      auto i = item.get_global_id();
      if (item.get_global_linear_id() == 0){
        out << " | get_group_range(): " << item.get_group_range() << "\n"
	    << " | get_global_range(): " << item.get_global_range() << "\n"
            << " | get_local_range(): " << item.get_local_range() << "\n"
	    << " | get_global_id()"
	    << " | get_global_linear_id()"
	    << " | get_local_id()"
            << " | get_local_linear_id()" 
	    << " | get_group_linear_id()"<< "\n";
      }

      out << " | " << item.get_global_id()
          << " | " << item.get_global_linear_id()
	  << " | " << item.get_local_id()
	  << " | " << item.get_local_linear_id() 
	  << " | " << item.get_group_linear_id() << "\n";   
    });  
  }).wait();
}
