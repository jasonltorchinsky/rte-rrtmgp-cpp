if(USEMPI)
  set(ENV{CC}  mpicc ) # C compiler for parallel build
  set(ENV{CXX} mpicxx) # C++ compiler for parallel build
  set(ENV{FC}  mpif90) # Fortran compiler for parallel build
else()
  set(ENV{CC}  gcc) # C compiler for serial build
  set(ENV{CXX} g++) # C++ compiler for serial build
  set(ENV{FC}  gfortran) # Fortran compiler for serial build
endif()

set(USER_CXX_FLAGS "-std=c++14 -DBOOL_TYPE=\"signed char\"")
set(USER_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")
set(USER_FC_FLAGS "-std=f2003 -fdefault-real-8 -fdefault-double-8 -fPIC -ffixed-line-length-none -fno-range-check")
set(USER_FC_FLAGS_RELEASE "-O3 -DNDEBUG")
set(USER_FC_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")

set(NETCDF_INCLUDE_DIR "/projects/aue/cee/builds/x86_64/rhel8/2410comp/toolchain-intel-2024.1.0/install/linux-rhel8-x86_64/oneapi-2024.1.0/netcdf-c-4.9.2-cools22/include")
set(NETCDF_LIB_C       "/projects/aue/cee/builds/x86_64/rhel8/2410comp/toolchain-intel-2024.1.0/install/linux-rhel8-x86_64/oneapi-2024.1.0/netcdf-c-4.9.2-cools22/lib/libnetcdf.so")

set(LIBS ${NETCDF_LIB_C})
set(INCLUDE_DIRS ${NETCDF_INCLUDE_DIR})

add_definitions(-DRESTRICTKEYWORD=__restrict__)
add_definitions(-DRTE_USE_CBOOL)
if(USECUDA)
  set(CMAKE_CUDA_ARCHITECTURES 70)
  set(CUDA_INCLUDE_DIR "/projects/aue/cee/deploy/4a1675f4/linux-rhel8-x86_64/none-none/cuda-12.9.0-dywi3s7/include")
  set(INCLUDE_DIRS ${INCLUDE_DIRS} ${CUDA_INCLUDE_DIR})

  ## find_package not used throuhgout this project, so we have to do it the hard way
  add_library(curand INTERFACE IMPORTED)
  set_target_properties(curand PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIR}
    INTERFACE_LINK_LIBRARIES "/projects/aue/cee/deploy/4a1675f4/linux-rhel8-x86_64/none-none/cuda-12.9.0-dywi3s7/lib64/libcurand.so")
endif()

set(CMAKE_DISABLE_FIND_PACKAGE_Git TRUE)
