#!/bin/bash
##---------------------------------------------------------------------------##
## CONFIGURE MCLS with CMake
##---------------------------------------------------------------------------##

EXTRA_ARGS=$@

rm -rf CMakeCache.txt

##---------------------------------------------------------------------------##

cmake \
    -D CMAKE_INSTALL_PREFIX:PATH=$PWD \
    -D CMAKE_BUILD_TYPE:STRING=DEBUG \
    -D CMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
    -D CMAKE_SKIP_RULE_DEPENDENCY:BOOL=ON \
    -D TPL_ENABLE_MPI:BOOL=ON \
    -D MPI_BASE_DIR:PATH=/local.hd/cnergg/stuart/builds/openmpi-1.4.6 \
    -D TPL_ENABLE_Boost:BOOL=ON \
    -D Boost_INCLUDE_DIRS:PATH=/local.hd/cnergg/stuart/builds/boost_1_51_0/include \
    -D TPL_ENABLE_BoostLib:BOOL=ON \
    -D BoostLib_INCLUDE_DIRS:PATH=/local.hd/cnergg/stuart/builds/boost_1_51_0/include \
    -D BoostLib_LIBRARY_DIRS:PATH=/local.hd/cnergg/stuart/builds/boost_1_51_0/lib \
    -D SPRNG_LIBRARY_DIRS:PATH=/local.hd/cnergg/stuart/software/sprng-0.5x \
    -D SPRNG_INCLUDE_DIRS:PATH=/local.hd/cnergg/stuart/software/sprng-0.5x/SRC \
    -D BLAS_LIBRARY_DIRS:PATH=/local.hd/cnergg/stuart/software/lapack-3.4.1 \
    -D BLAS_LIBRARY_NAMES:STRING="blas" \
    -D LAPACK_LIBRARY_DIRS:PATH=/local.hd/cnergg/stuart/software/lapack-3.4.1 \
    -D LAPACK_LIBRARY_NAMES:STRING="lapack" \
    -D Trilinos_EXTRA_REPOSITORIES="MCLS" \
    -D Trilinos_ENABLE_ALL_PACKAGES:BOOL=OFF \
    -D Trilinos_ENABLE_ALL_OPTIONAL_PACKAGES:BOOL=ON \
    -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
    -D Trilinos_ENABLE_TESTS:BOOL=OFF \
    -D Trilinos_ENABLE_DEBUG:BOOL=ON \
    -D Trilinos_ENABLE_INSTALL_CMAKE_CONFIG_FILES:BOOL=OFF \
    -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
    -D Trilinos_ENABLE_MCLS:BOOL=ON \
    -D MCLS_ENABLE_TESTS:BOOL=ON \
    -D MCLS_ENABLE_Tpetra:BOOL=ON \
    -D MCLS_ENABLE_Epetra:BOOL=ON \
    -D MCLS_ENABLE_EpetraExt:BOOL=ON \
    -D MCLS_ENABLE_Thyra:BOOL=ON \
    -D MCLS_ENABLE_Stratimikos:BOOL=ON \
    -D MCLS_ENABLE_DBC:BOOL=ON \
    $EXTRA_ARGS \
    /local.hd/cnergg/stuart/software/Trilinos
