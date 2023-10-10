// SYCL, buffer accessor properties

#include <sycl/sycl.hpp>

constexpr int N = 1024;

int main() {

  std::vector<int> a(N, 2);
  std::vector<int> b(N, 3);
  std::vector<int> c(N);
    
  // create a sycl queue with GPU device
  sycl::queue q (sycl::gpu_selector_v);
  std::cout << "Offload Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  {
    sycl::buffer<int> a_buf(a);
    sycl::buffer<int> b_buf(b);
    sycl::buffer<int> c_buf(c);

    q.submit([&](auto &h) {
      // Create accessors with access modes and properties
      sycl::accessor a_device(a_buf, h, sycl::read_only);
      sycl::accessor b_device(b_buf, h, sycl::read_only);
      sycl::accessor c_device(c_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(N, [=](auto i) {
        c_device[i] = a_device[i] + b_device[i];
      });
    });
  }

  // print output
  for (int i=0; i<N; i++) std::cout << c[i] << " ";
  std::cout << "\n";
}
