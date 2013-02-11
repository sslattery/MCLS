INCLUDE(TribitsAddExecutableAndTest)
INCLUDE(TribitsCopyFilesToBinaryDir)

INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_BINARY_DIR}/../../src)

TRIBITS_ADD_EXECUTABLE(
  test_single_mcls_thyra_solver_driver
  SOURCES
    test_single_mcls_thyra_solver_driver.cpp
    test_single_mcls_thyra_solver.cpp
  COMM serial mpi
  )

TRIBITS_ADD_TEST(
  test_single_mcls_thyra_solver_driver
  POSTFIX_AND_ARGS_0 FourByFour
    --max-iters=4 --no-use-preconditioner --show-all-tests  --matrix-file=FourByFour.mtx
  POSTFIX_AND_ARGS_1 nos5_kl190
    --show-all-tests --gmres-krylov-length=190 --max-solution-error=1e-2 --max-resid=1e-6
    --matrix-file=nos5.mtx
  POSTFIX_AND_ARGS_2 nos1_np
    --no-use-preconditioner --show-all-tests --num-rhs=1 --block-size=1
    --gmres-krylov-length=300 --max-fwd-error=1e-13 --max-resid=1e-13 --max-iters=1000
    --max-solution-error=1e-6 --matrix-file=nos1.mtx
  POSTFIX_AND_ARGS_3 nos1
    --show-all-tests --num-rhs=1 --block-size=1 --gmres-krylov-length=300
    --max-fwd-error=1e-13 --max-resid=1e-13 --max-iters=1000 --max-solution-error=1e-6
    --matrix-file=nos1.mtx
  POSTFIX_AND_ARGS_4 nos1_nrhs8
    --show-all-tests --num-rhs=8 --block-size=8 --gmres-krylov-length=300
    --max-fwd-error=1e-13 --max-resid=1e-13 --max-iters=1000 --max-solution-error=1e-6
    --matrix-file=nos1.mtx
  POSTFIX_AND_ARGS_0 nos1_bs8
    --no-use-preconditioner --show-all-tests --num-rhs=8 --block-size=8
    --gmres-krylov-length=300 --max-fwd-error=1e-13 --max-resid=1e-13
    --max-iters=1000 --max-solution-error=1e-6
    --matrix-file=nos1.mtx
  COMM serial mpi
  NUM_MPI_PROCS 4
  )

TRIBITS_COPY_FILES_TO_BINARY_DIR(CopyMclsTestFiles
  DEST_FILES
    bcsstk01.mtx                              
    bcsstk02.mtx                              
    bcsstk04.mtx                              
    nos1.mtx                                      
    nos5.mtx                                      
    FourByFour.mtx                          
    Diagonal.mtx                              
    MissingADiagonal.mtx                              
    KheadK.mtx                                  
    KheadSorted.mtx
    Khead.triS  
  SOURCE_PREFIX
    "In_"
  SOURCE_DIR
    "${PACKAGE_SOURCE_DIR}/../amesos/test/Test_Basic"
  EXEDEPS
    test_single_mcls_thyra_solver_driver
  )
