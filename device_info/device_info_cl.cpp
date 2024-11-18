// Print Intel GPU Device Info using OpenCL Backend
// oneAPI 2025.0 compiler
// icpx -fsycl device_info_cl.cpp -lOpenCL
// ONEAPI_DEVICE_SELECTOR=opencl:gpu; ./a.out

#include <sycl/sycl.hpp>
//#include <CL/cl.h>
//#include <CL/cl_ext.h>

int main() {
    try {
        // Create a SYCL queue
        sycl::queue sycl_queue(sycl::gpu_selector_v);
        auto sycl_device = sycl_queue.get_device();

        // Query the Intel-specific OpenCL property for sub-slice count
        auto cl_device = sycl::get_native<sycl::backend::opencl>(sycl_device);
        cl_uint slice_count = 0, sub_slice_per_slice_count = 0, eu_per_sub_slice_count = 0, threads_per_eu_count = 0, feature_capabilities = 0;
        clGetDeviceInfo(cl_device, CL_DEVICE_NUM_SLICES_INTEL, sizeof(cl_uint), &slice_count, nullptr);
        clGetDeviceInfo(cl_device, CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL, sizeof(cl_uint), &sub_slice_per_slice_count, nullptr);
        clGetDeviceInfo(cl_device, CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL, sizeof(cl_uint), &eu_per_sub_slice_count, nullptr);
        clGetDeviceInfo(cl_device, CL_DEVICE_NUM_THREADS_PER_EU_INTEL, sizeof(cl_uint), &threads_per_eu_count, nullptr);
        clGetDeviceInfo(cl_device, CL_DEVICE_FEATURE_CAPABILITIES_INTEL, sizeof(cl_uint), &feature_capabilities, nullptr);

        // Print device_info
        std::cout << "\n opencl Device Info:\n";

        std::cout << "\n Device Name: " << sycl_device.get_info<sycl::info::device::name>() << "\n";

        std::cout << "\n slice_count: " << slice_count;
        std::cout << "\n sub_slice_per_slice_count: " << sub_slice_per_slice_count;
        std::cout << "\n eu_per_sub_slice_count: " << eu_per_sub_slice_count;
        std::cout << "\n threads_per_eu_count: " << threads_per_eu_count;
        
        std::cout << "\n\n Total Sub-slice count: " << slice_count * sub_slice_per_slice_count;
        std::cout << "\n Total EU count: " << slice_count * sub_slice_per_slice_count * eu_per_sub_slice_count;
        std::cout << "\n Total HW Threads count: " << slice_count * sub_slice_per_slice_count * eu_per_sub_slice_count * threads_per_eu_count;
        std::cout << "\n\n";
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}

