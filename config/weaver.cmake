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
set(USER_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")
set(USER_FC_FLAGS "-std=f2003 -fdefault-real-8 -fdefault-double-8 -fPIC -ffixed-line-length-none -fno-range-check")
set(USER_FC_FLAGS_RELEASE "-O2 -DNDEBUG")
set(USER_FC_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")
 
set(NETCDF_INCLUDE_DIR "/projects/ppc64le-pwr9-rhel8/tpls/netcdf-c/4.9.2/gcc/12.2.0/openmpi/4.1.6/xoi6woi/include")
set(NETCDF_LIB_C       "/projects/ppc64le-pwr9-rhel8/tpls/netcdf-c/4.9.2/gcc/12.2.0/openmpi/4.1.6/xoi6woi/lib/libnetcdf.so")
 
set(LIBS ${NETCDF_LIB_C}) # ${HDF5_LIB_2} ${HDF5_LIB_1} m z curl)
set(INCLUDE_DIRS ${FFTW_INCLUDE_DIR} ${NETCDF_INCLUDE_DIR})
 
 
add_definitions(-DRESTRICTKEYWORD=__restrict__)
add_definitions(-DRTE_USE_CBOOL)
if(USECUDA)
  set(CMAKE_CUDA_ARCHITECTURES 70)
  set(CMAKE_CUDA_FLAGS "${MAKE_CUDA_FLAGS} -Xptxas -v")
endif()
