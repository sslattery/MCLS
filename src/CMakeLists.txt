INCLUDE(TribitsLibraryMacros)

#
# A) Package-specific configuration options
#

TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_config.hpp)

#
# B) Define the header and source files (and directories)
#

SET(HEADERS "")
SET(SOURCES "")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

SET(HEADERS ${HEADERS}
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.hpp
  )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

APPEND_SET(HEADERS
  MCLS_AdjointHistory.hpp
  MCLS_AdjointHistory_impl.hpp
  MCLS_AdjointTally.hpp
  MCLS_AdjointTally_impl.hpp
  MCLS_AlmostOptimalDomain.hpp
  MCLS_AlmostOptimalDomain_impl.hpp
  MCLS_AndersonSolverManager.hpp
  MCLS_AndersonSolverManager_impl.hpp
  MCLS_CommHistoryBuffer.hpp
  MCLS_CommHistoryBuffer_impl.hpp
  MCLS_CommTools.hpp
  MCLS_DBC.hpp
  MCLS_DomainCommunicator.hpp
  MCLS_DomainCommunicator_impl.hpp
  MCLS_DomainTraits.hpp
  MCLS_DomainTransporter.hpp
  MCLS_DomainTransporter_impl.hpp
  MCLS_Events.hpp
  MCLS_FixedPointIteration.hpp
  MCLS_FixedPointIterationFactory.hpp
  MCLS_FixedPointIterationFactory_impl.hpp
  MCLS_FixedPointSolverManager.hpp
  MCLS_FixedPointSolverManager_impl.hpp
  MCLS_ForwardHistory.hpp
  MCLS_ForwardHistory_impl.hpp
  MCLS_ForwardTally.hpp
  MCLS_ForwardTally_impl.hpp
  MCLS_GlobalTransporter.hpp
  MCLS_GlobalTransporterFactory.hpp
  MCLS_GlobalTransporterFactory_impl.hpp
  MCLS_History.hpp
  MCLS_History_impl.hpp
  MCLS_HistoryBuffer.hpp
  MCLS_HistoryBuffer_impl.hpp
  MCLS_HistoryTraits.hpp
  MCLS_LinearProblem.hpp
  MCLS_LinearProblem_impl.hpp
  MCLS_MatrixTraits.hpp
  MCLS_MCSAModelEvaluator.hpp
  MCLS_MCSAModelEvaluator_impl.hpp
  MCLS_MCSASolverManager.hpp
  MCLS_MCSASolverManager_impl.hpp
  MCLS_MCSAStatusTest.hpp
  MCLS_MCSAStatusTest_impl.hpp
  MCLS_MCSolver.hpp
  MCLS_MCSolver_impl.hpp
  MCLS_MonteCarloSolverManager.hpp
  MCLS_MonteCarloSolverManager_impl.hpp
  MCLS_MinimalResidualIteration.hpp
  MCLS_MinimalResidualIteration_impl.hpp
  MCLS_Preconditioner.hpp
  MCLS_PRNG.hpp
  MCLS_PRNG_impl.hpp
  MCLS_RNGTraits.hpp
  MCLS_RNSDIteration.hpp
  MCLS_RNSDIteration_impl.hpp
  MCLS_RichardsonIteration.hpp
  MCLS_RichardsonIteration_impl.hpp
  MCLS_SamplingTools.hpp
  MCLS_Serializer.hpp
  MCLS_SolverFactory.hpp
  MCLS_SolverFactory_impl.hpp
  MCLS_SourceTraits.hpp
  MCLS_SourceTransporter.hpp
  MCLS_SourceTransporter_impl.hpp
  MCLS_SteepestDescentIteration.hpp
  MCLS_SteepestDescentIteration_impl.hpp
  MCLS_SubdomainTransporter.hpp
  MCLS_SubdomainTransporter_impl.hpp
  MCLS_TallyTraits.hpp
  MCLS_ThyraVectorExtraction.hpp
  MCLS_UniformAdjointSource.hpp
  MCLS_UniformAdjointSource_impl.hpp
  MCLS_UniformForwardSource.hpp
  MCLS_UniformForwardSource_impl.hpp
  MCLS_VectorExport.hpp
  MCLS_VectorTraits.hpp
  MCLS_Xorshift.hpp
  ) 

APPEND_SET(SOURCES
  MCLS_CommTools.cpp
  MCLS_DBC.cpp
  )

#
# C) Define the targets for package's library(s)
#

TRIBITS_ADD_LIBRARY(
  mcls	
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  )

INSTALL(FILES 
  ${HEADERS} 
  DESTINATION 
  ${CMAKE_INSTALL_PREFIX}/include/
)
