// Print Intel GPU Device Info
// oneAPI 2025.0 compiler
// icpx -fsycl device_info_sycl.cpp

#include <sycl/sycl.hpp>
int main() {
  sycl::queue q(sycl::gpu_selector_v);

  std::cout << "device name        : " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "global_mem_size    : " << q.get_device().get_info<sycl::info::device::global_mem_size>() << "\n";
  std::cout << "local_mem_size     : " << q.get_device().get_info<sycl::info::device::local_mem_size>() << "\n";
  std::cout << "max_mem_alloc_size : " << q.get_device().get_info<sycl::info::device::max_mem_alloc_size>() << "\n";
  std::cout << "max_work_group_size: " << q.get_device().get_info<sycl::info::device::max_work_group_size>() << "\n";
  std::cout << "max_compute_units  : " << q.get_device().get_info<sycl::info::device::max_compute_units>() << "\n";
}
