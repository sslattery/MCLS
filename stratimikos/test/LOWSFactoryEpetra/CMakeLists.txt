INCLUDE(TribitsAddExecutableAndTest)
INCLUDE(TribitsCopyFilesToBinaryDir)

INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_BINARY_DIR}/../../src)

TRIBITS_ADD_EXECUTABLE(
  test_single_mcls_stratimikos_solver_driver
  SOURCES
    test_single_mcls_stratimikos_solver_driver.cpp
    test_single_mcls_stratimikos_solver.cpp
  COMM serial mpi
  )

TRIBITS_ADD_TEST(
  test_single_mcls_stratimikos_solver_driver
  POSTFIX_AND_ARGS_0 FourByFour_mcsa_nrhs1
    --max-iters=15 --no-use-preconditioner --show-all-tests --matrix-file=FourByFour.mtx
    --mc-histories=1000 --solver-type="MCSA" --num-rhs=1
  POSTFIX_AND_ARGS_1 FourByFour_mcsa_nrhs8
    --max-iters=15 --no-use-preconditioner --show-all-tests --matrix-file=FourByFour.mtx
    --mc-histories=1000 --solver-type="MCSA" --num-rhs=8 
  POSTFIX_AND_ARGS_2 FourByFour_seqmc_nrhs1
    --max-iters=100 --no-use-preconditioner --show-all-tests --matrix-file=FourByFour.mtx
    --mc-histories=1000 --solver-type="Sequential MC"
  POSTFIX_AND_ARGS_3 FourByFour_seqmc_nrhs2
    --max-iters=100 --no-use-preconditioner --show-all-tests --matrix-file=FourByFour.mtx
    --mc-histories=1000 --solver-type="Sequential MC" --num-rhs=2
  POSTFIX_AND_ARGS_4 KheadK_adjmc_nrhs1
    --use-preconditioner --show-all-tests --num-rhs=1
    --max-fwd-error=1e-2 --max-resid=1e-2 --max-solution-error=1e-2 --max-iters=1
    --mc-histories=100000 --matrix-file=KheadK.mtx
     --solver-type="Adjoint MC" --prec-type="Block Jacobi" --block-size=2
  POSTFIX_AND_ARGS_5 KheadK_adjmc_nrhs3
    --use-preconditioner --show-all-tests --num-rhs=3
    --max-fwd-error=1e-2 --max-resid=1e-2 --max-solution-error=1e-2 --max-iters=1
    --mc-histories=100000 --matrix-file=KheadK.mtx
     --solver-type="Adjoint MC" --prec-type="Block Jacobi" --block-size=2
  POSTFIX_AND_ARGS_6 KheadK_mcsa_nrhs1
    --use-preconditioner --show-all-tests --num-rhs=1
    --max-fwd-error=1e-13 --max-resid=1e-13 --max-iters=1000 --mc-histories=100
    --max-solution-error=1e-6 --matrix-file=KheadK.mtx --solver-type="MCSA" 
    --prec-type="Point Jacobi"
  POSTFIX_AND_ARGS_7 KheadK_mcsa_nrhs8
    --use-preconditioner --show-all-tests --num-rhs=8
    --max-fwd-error=1e-13 --max-resid=1e-13 --max-iters=1000 --mc-histories=100
    --max-solution-error=1e-6 --matrix-file=KheadK.mtx --solver-type="MCSA" 
    --prec-type="Point Jacobi"
  POSTFIX_AND_ARGS_8 KheadK_seqmc_nrhs1
    --use-preconditioner --show-all-tests --num-rhs=1
    --max-fwd-error=1e-13 --max-resid=1e-13 --max-iters=1000 --mc-histories=100
    --max-solution-error=1e-6 --matrix-file=KheadK.mtx --solver-type="Sequential MC" 
    --prec-type="Point Jacobi"
  POSTFIX_AND_ARGS_9 KheadK_seqmc_nrhs2
    --use-preconditioner --show-all-tests --num-rhs=2
    --max-fwd-error=1e-13 --max-resid=1e-13 --max-iters=1000 --mc-histories=100
    --max-solution-error=1e-6 --matrix-file=KheadK.mtx --solver-type="Sequential MC" 
    --prec-type="Point Jacobi"  COMM serial mpi
  )

TRIBITS_COPY_FILES_TO_BINARY_DIR(CopyStratimikosTestFiles
  DEST_FILES
    FourByFour.mtx
    KheadK.mtx                                  
  SOURCE_PREFIX
    "In_"
  SOURCE_DIR
    "${Trilinos_SOURCE_DIR}/packages/amesos/test/Test_Basic"
  EXEDEPS
    test_single_mcls_stratimikos_solver_driver
  )
