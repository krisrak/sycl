// SYCL, print work-group and sub-group info and indexes 

#include <sycl/sycl.hpp>

int main(){
  // create a sycl queue with GPU device
  sycl::queue q (sycl::gpu_selector_v);
  std::cout << "Offload Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  
  auto N = 64; // global size
  auto B = 64; // work-group size

  // offload parallel compute on GPU using parallel_for
  q.submit([&](sycl::handler &h){
    // setup sycl stream class to print standard output from device code
    auto out = sycl::stream(2048, 2048, h);
    
    h.parallel_for(sycl::nd_range<1>(N, B), [=](sycl::nd_item<1> item){
      auto i = item.get_global_id();
      auto sg = item.get_sub_group();

      if (item.get_global_linear_id() == 0){
        out << " | get_group_range(): " << item.get_group_range() << "\n"
	    << " | get_global_range(): " << item.get_global_range() << "\n"
            << " | get_local_range(): " << item.get_local_range() << "\n"
	    << " | sg.get_local_range(): " << sg.get_local_range() << "\n"
	    << " | sg.get_group_range(): " << sg.get_group_range() << "\n"
	    << " | get_global_id()"
	    << " | get_global_linear_id()"
	    << " | get_local_id()"
            << " | get_local_linear_id()"
	    << " | get_group_linear_id()" 
	    << " | sg.get_group_id()" 
	    << " | sg.get_local_id()" << "\n";
      }

      out << " | " << item.get_global_id()
          << " | " << item.get_global_linear_id()
	  << " | " << item.get_local_id()
	  << " | " << item.get_local_linear_id()
          << " | " << item.get_group_linear_id() 
	  << " | " << sg.get_group_id() 
	  << " | " << sg.get_local_id() << "\n";	  
    });  
  }).wait();
}
