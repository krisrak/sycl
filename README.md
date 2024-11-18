# SYCL
SYCL is template library specification for C++ that enables programming applications to offload computations to heterogenous devices like GPU or FPGA from any vendor

## SYCL code examples

| Category | Directory | Sample | Description
|---|---|---|---
|Device Selection| [device](device)|  [device.cpp](device/device.cpp) | create sycl::queue with device selection
||| [gpus.cpp](device/gpus.cpp) | get multiple gpus on system
||| [all_devices.cpp](device/all_devices.cpp) | get all avialable devices on system
|Compute Offload| [offload](offload) | [cpp.cpp](offload/cpp.cpp) | C++ example of computation
||| [single_task.cpp](offload/single_task.cpp) | Offload computation using single_task
||| [parallel_for.cpp](offload/parallel_for.cpp) | Offload parallel computation using parallel_for
|Memory Managment| [usm](usm) | [malloc_host.cpp](usm/malloc_host.cpp) | Allocation memory on host and access on device
||| [malloc_shared.cpp](usm/malloc_shared.cpp) | Allocate memory that migrated between host and device
||| [malloc_device.cpp](usm/malloc_device.cpp) | Allocate memory on device and manually copy memory between host and device
||| [usm_in_order_queue.cpp](usm/usm_in_order_queue.cpp) | Use in_order queue property to set dependency between kernels
||| [usm_event_dependency.cpp](usm/usm_event_dependency.cpp) | Use kernel events for kernel execution dependency
|| [buffer](buffer) | [buffer.cpp](buffer/buffer.cpp) | Buffer Accessor memory model usage
||| [buffer_destruction.cpp](buffer/buffer_destruction.cpp) | Buffer Accessor memory model with buffer destruction
||| [buffer_accessor_properties.cpp](buffer/buffer_accessor_properties.cpp) | Buffer Accessor memory model with accessor modes and properties
|Kernel Programming| [group](group) | [nd_range.cpp](group/nd_range.cpp) | nd_range kernel for grouped parallel execution on device 
||| [work_group_info.cpp](group/work_group_info.cpp) | print global and local indexes for all work-items
||| [work_group_info_3d.cpp](group/work_group_info_3d.cpp) | print global and local indexes for all 3d work-items
||| [work_group_reduce.cpp](group/work_group_reduce.cpp) | reduction group algorithm on work-group
||| [sub_group_info.cpp](group/sub_group_info.cpp) | print global and local indexes, sub-group index for all work-items
||| [sub_group_size.cpp](group/sub_group_size.cpp) | set specific sub-group size
||| [sub_group_reduce.cpp](group/sub_group_reduce.cpp) | reduction group algorithm on sub-group
||| [sub_group_shuffle.cpp](group/sub_group_shuffle.cpp) | shuffle group algorithm on sub-group
||| [local_mem.cpp](group/local_mem.cpp) | access local memory on device for faster computation
|Device Info| [device_info](device_info)|  [device_info_sycl.cpp](device_info/device_info_sycl.cpp) | Get GPU device info using SYCL
||| [device_info_cl.cpp](device_info/device_info_cl.cpp) | Get extra GPU device info using OpenCL backend
||| [device_info_ze.cpp](device_info/device_info_ze.cpp) | Get extra GPU device info using Level-Zero backend

## Compiling SYCL examples
- Download and Install C++ SYCL Compiler: [Intel oneAPI C++/DPC++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
- Initialize environment varialbles: `source /opt/intel/oneapi/setvars.sh`
- Compile: `icpx -fsycl test.cpp`
- Run: `./a.out`
