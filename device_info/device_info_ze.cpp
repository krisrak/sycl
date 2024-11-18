// Print Intel GPU Device Info using Level-Zero Backend
// oneAPI 2025.0 compiler
// Install development packages: sudo apt install -y libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev level-zero-dev
// icpx -fsycl device_info_ze.cpp -lze_loader

#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

int main() {
    try {
        // Create a SYCL queue
        sycl::queue sycl_queue(sycl::gpu_selector_v);
        auto sycl_device = sycl_queue.get_device();

        // Get level_zero device properties
        auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_device);
        ze_device_properties_t device_properties;
        zeDeviceGetProperties(ze_device, &device_properties);

        // Print device_info
        std::cout << "\n level_zero Device Info:\n";
        
        std::cout << "\n Device Name: " << device_properties.name;
        
        std::cout << "\n\n numSlices: " << device_properties.numSlices;
        std::cout << "\n numSubslicesPerSlice: " << device_properties.numSubslicesPerSlice;
        std::cout << "\n numEUsPerSubslice: " << device_properties.numEUsPerSubslice;
        std::cout << "\n numThreadsPerEU: " << device_properties.numThreadsPerEU;
        
        std::cout << "\n\n Total Sub-slice count: " << device_properties.numSlices * device_properties.numSubslicesPerSlice;
        std::cout << "\n Total EU count: " << device_properties.numSlices * device_properties.numSubslicesPerSlice * device_properties.numEUsPerSubslice;
        std::cout << "\n Total HW Threads count: " << device_properties.numSlices * device_properties.numSubslicesPerSlice * device_properties.numEUsPerSubslice * device_properties.numThreadsPerEU;
        std::cout << "\n\n";
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}

