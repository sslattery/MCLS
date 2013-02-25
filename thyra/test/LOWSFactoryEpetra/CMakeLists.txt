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
    --max-iters=15 --no-use-preconditioner --show-all-tests --matrix-file=FourByFour.mtx
    --mc-histories=1000
  POSTFIX_AND_ARGS_1 nos5_kl190
    --show-all-tests --no-use-preconditioner --max-solution-error=1e-2 --max-resid=1e-6
    --matrix-file=nos5.mtx --use-scaling --mc-histories=1000
  POSTFIX_AND_ARGS_2 nos1_np
    --no-use-preconditioner --show-all-tests --num-rhs=1
    --max-fwd-error=1e-13 --max-resid=1e-13 --max-iters=1000 --mc-histories=100
    --max-solution-error=1e-6 --matrix-file=nos1.mtx --use-scaling
  POSTFIX_AND_ARGS_3 nos1_nrhs8
    --no-use-preconditioner --show-all-tests --num-rhs=8 --mc-histories=100
    --max-fwd-error=1e-13 --max-resid=1e-13 --max-iters=1000 --max-solution-error=1e-6
    --matrix-file=nos1.mtx --use-scaling
  POSTFIX_AND_ARGS_4 nos1_bs8
    --no-use-preconditioner --show-all-tests --num-rhs=8 --mc-histories=100
    --max-fwd-error=1e-13 --max-resid=1e-13
    --max-iters=1000 --max-solution-error=1e-6
    --matrix-file=nos1.mtx --use-scaling
  COMM serial mpi
  NUM_MPI_PROCS 1
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
    "${PACKAGE_SOURCE_DIR}/../packages/amesos/test/Test_Basic"
  EXEDEPS
    test_single_mcls_thyra_solver_driver
  )
