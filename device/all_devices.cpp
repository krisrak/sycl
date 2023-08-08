#include <sycl/sycl.hpp>

int main(){
  for(auto &p : sycl::platform::get_platforms()){
    std::cout << "Platform: " << p.get_info<sycl::info::platform::name>() << "\n";
    for(auto &d : p.get_devices()){
      std::cout << " -Device: " << d.get_info<sycl::info::device::name>() << "\n";
    }
  }
}
