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

set(NETCDF_INCLUDE_DIR "/opt/cray/pe/netcdf/4.9.0.9/gnu/12.3/include/")
set(NETCDF_LIB_C       "/opt/cray/pe/netcdf/4.9.0.9/gnu/12.3/lib/libnetcdf.so")

set(LIBS ${NETCDF_LIB_C})
set(INCLUDE_DIRS ${NETCDF_INCLUDE_DIR})

add_definitions(-DRESTRICTKEYWORD=__restrict__)
add_definitions(-DRTE_USE_CBOOL)
if(USECUDA)
  set(CMAKE_CUDA_ARCHITECTURES 70)
  set(CUDA_INCLUDE_DIR "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4/include")
  set(INCLUDE_DIRS ${INCLUDE_DIRS} ${CUDA_INCLUDE_DIR})

  ## find_package not used throuhgout this project, so we have to do it the hard way
  add_library(curand INTERFACE IMPORTED)
  set_target_properties(curand PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIR}
    INTERFACE_LINK_LIBRARIES "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4/lib64/libcurand.so")
endif()

set(CMAKE_DISABLE_FIND_PACKAGE_Git TRUE)
