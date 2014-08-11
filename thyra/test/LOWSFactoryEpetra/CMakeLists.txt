INCLUDE(TribitsAddExecutableAndTest)
INCLUDE(TribitsCopyFilesToBinaryDir)

INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_BINARY_DIR}/../../src)

TRIBITS_ADD_EXECUTABLE(
  test_single_mcls_thyra_solver_driver_epetra
  SOURCES
    test_single_mcls_thyra_solver_driver.cpp
    test_single_mcls_thyra_solver.cpp
  COMM serial mpi
  )

TRIBITS_ADD_TEST(
  test_single_mcls_thyra_solver_driver_epetra
  NUM_MPI_PROCS 1
  POSTFIX_AND_ARGS_0 FourByFour_mcsa_nrhs1
    --max-iters=15 --no-use-preconditioner --show-all-tests 
    --matrix-file=FourByFour.mtx
    --mc-histories=1000 --solver-type="MCSA" --num-rhs=1
  POSTFIX_AND_ARGS_1 FourByFour_mcsa_nrhs8
    --max-iters=15 --no-use-preconditioner --show-all-tests 
    --matrix-file=FourByFour.mtx
    --mc-histories=1000 --solver-type="MCSA" --num-rhs=8 
  POSTFIX_AND_ARGS_2 FourByFour_richardson_nrhs3
    --max-iters=1000 --no-use-preconditioner --show-all-tests 
    --matrix-file=FourByFour.mtx --rich-relax=0.9
    --solver-type="Fixed Point" --num-rhs=3
  POSTFIX_AND_ARGS_3 KheadK_adjmc_nrhs1
    --use-preconditioner --show-all-tests --num-rhs=1
    --max-fwd-error=1e-2 --max-resid=1e-2 --max-solution-error=1e-2 --max-iters=1
    --mc-histories=100000 --matrix-file=KheadK.mtx
     --solver-type="Adjoint MC" --prec-type="Block Jacobi" --block-size=2
  POSTFIX_AND_ARGS_4 KheadK_adjmc_nrhs2
    --use-preconditioner --show-all-tests --num-rhs=2
    --max-fwd-error=1e-2 --max-resid=1e-2 --max-solution-error=1e-2 --max-iters=1
    --mc-histories=100000 --matrix-file=KheadK.mtx
     --solver-type="Adjoint MC" --prec-type="Block Jacobi" --block-size=2
  POSTFIX_AND_ARGS_5 KheadK_mcsa_nrhs1
    --use-preconditioner --show-all-tests --num-rhs=1
    --max-fwd-error=1e-13 --max-resid=1e-13 --max-iters=1000 --mc-histories=100
    --max-solution-error=1e-6 --matrix-file=KheadK.mtx --solver-type="MCSA" 
    --prec-type="Point Jacobi"
  POSTFIX_AND_ARGS_6 KheadK_mcsa_nrhs8
    --use-preconditioner --show-all-tests --num-rhs=8
    --max-fwd-error=1e-13 --max-resid=1e-13 --max-iters=1000 --mc-histories=100
    --max-solution-error=1e-6 --matrix-file=KheadK.mtx --solver-type="MCSA" 
    --prec-type="Point Jacobi"
  POSTFIX_AND_ARGS_7 KheadK_richardson_nrhs1
    --use-preconditioner --show-all-tests --num-rhs=1
    --max-fwd-error=1e-13 --max-resid=1e-13 --max-iters=1000
    --max-solution-error=1e-6 --matrix-file=KheadK.mtx --solver-type="Fixed Point" 
    --prec-type="Point Jacobi"
  COMM serial mpi
  )

TRIBITS_COPY_FILES_TO_BINARY_DIR(CopyMclsTestFilesEpetra
  DEST_FILES
    FourByFour.mtx
    KheadK.mtx
    bcsstk04.mtx
  SOURCE_PREFIX
    "In_"
  SOURCE_DIR
    "${Trilinos_SOURCE_DIR}/packages/amesos/test/Test_Basic"
  EXEDEPS
    test_single_mcls_thyra_solver_driver_epetra
  )
