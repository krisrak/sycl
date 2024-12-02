// Print Intel GPU Device Info
// oneAPI 2025.0 compiler
// icpx -fsycl device_info_sycl.cpp

#include <sycl/sycl.hpp>
int main() {
  sycl::queue q(sycl::gpu_selector_v);

  std::cout << "device name               : " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::cout << "local_mem_size            : " << q.get_device().get_info<sycl::info::device::local_mem_size>() << "\n";
  std::cout << "global_mem_size           : " << q.get_device().get_info<sycl::info::device::global_mem_size>() << "\n";
  std::cout << "max_mem_alloc_size        : " << q.get_device().get_info<sycl::info::device::max_mem_alloc_size>() << "\n";
  std::cout << "max_work_group_size       : " << q.get_device().get_info<sycl::info::device::max_work_group_size>() << "\n";
  std::cout << "max_compute_units         : " << q.get_device().get_info<sycl::info::device::max_compute_units>() << "\n";
  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  std::cout << "sub_group_sizes           : ";
  for (int i=0; i<sg_sizes.size(); i++) std::cout << sg_sizes[i] << " "; std::cout << "\n";

  auto numSlices = q.get_device().get_info<sycl::ext::intel::info::device::gpu_slices>();
  auto numSubslicesPerSlice = q.get_device().get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
  auto numEUsPerSubslice = q.get_device().get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>();
  auto numThreadsPerEU = q.get_device().get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();
  std::cout << "gpu_slices                : " << numSlices << "\n";
  std::cout << "gpu_subslices_per_slice   : " << numSubslicesPerSlice << "\n";
  std::cout << "gpu_eu_count_per_subslice : " << numEUsPerSubslice << "\n";
  std::cout << "gpu_hw_threads_per_eu     : " << numThreadsPerEU << "\n";
  std::cout << " XeCore count             : " << numSlices * numSubslicesPerSlice << "\n";
  std::cout << " EU count                 : " << numSlices * numSubslicesPerSlice * numEUsPerSubslice << "\n";
  std::cout << " EU Threads count         : " << numSlices * numSubslicesPerSlice * numEUsPerSubslice * numThreadsPerEU << "\n";

}
