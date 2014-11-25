INCLUDE(TribitsAddExecutableAndTest)
INCLUDE(TribitsAddAdvancedTest)

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraVector_tests
  SOURCES tstTpetraVector.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraVectorExport_tests
  SOURCES tstTpetraVectorExport.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraHelpers_tests
  SOURCES tstTpetraHelpers.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraCrsMatrix_tests
  SOURCES tstTpetraCrsMatrix.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraLinearProblem_tests
  SOURCES tstTpetraLinearProblem.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraAdjointTally_tests
  SOURCES tstTpetraAdjointTally.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraForwardTally_tests
  SOURCES tstTpetraForwardTally.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraAdjointDomain_tests
  SOURCES tstTpetraAdjointDomain.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraForwardDomain_tests
  SOURCES tstTpetraForwardDomain.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraDomainTransporter_tests
  SOURCES tstTpetraDomainTransporter.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraDomainCommunicator_tests
  SOURCES tstTpetraDomainCommunicator.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraUniformAdjointSource_tests
  SOURCES tstTpetraUniformAdjointSource.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
   STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraUniformForwardSource_tests
  SOURCES tstTpetraUniformForwardSource.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
   STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraSourceTransporter_tests
  SOURCES tstTpetraSourceTransporter.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraMSODManager_tests
  SOURCES tstTpetraMSODManager.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraMCSolver_tests
  SOURCES tstTpetraMCSolver.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraAdjointSolverManager_tests
  SOURCES tstTpetraAdjointSolverManager.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraForwardSolverManager_tests
  SOURCES tstTpetraForwardSolverManager.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraMCSASolverManager_tests
  SOURCES tstTpetraMCSASolverManager.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraFixedPointSolverManager_tests
  SOURCES tstTpetraFixedPointSolverManager.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraSolverFactory_tests
  SOURCES tstTpetraSolverFactory.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraPointJacobiPreconditioner_tests
  SOURCES tstTpetraPointJacobiPreconditioner.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  TpetraBlockJacobiPreconditioner_tests
  SOURCES tstTpetraBlockJacobiPreconditioner.cpp ${TEUCHOS_STD_PARALLEL_UNIT_TEST_MAIN}
  COMM serial mpi
  STANDARD_PASS_OUTPUT
  )
