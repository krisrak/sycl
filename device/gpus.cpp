#include <sycl/sycl.hpp>

int main(){
  auto gpus = sycl::platform(sycl::gpu_selector_v).get_devices();
  for(auto &gpu : gpus){
    std::cout << "Device: " << gpu.get_info<sycl::info::device::name>() << "(" << gpu.get_platform().get_info<sycl::info::platform::name>()  << ")\n";
  }
}
