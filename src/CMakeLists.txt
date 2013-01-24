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
  MCLS_AdjointDomain.hpp
  MCLS_AdjointDomain_impl.hpp
  MCLS_AdjointTally.hpp
  MCLS_AdjointTally_impl.hpp
  MCLS_CommHistoryBuffer.hpp
  MCLS_CommHistoryBuffer_impl.hpp
  MCLS_CommTools.hpp
  MCLS_DBC.hpp
  MCLS_DomainCommunicator.hpp
  MCLS_DomainCommunicator_impl.hpp
  MCLS_DomainTransporter.hpp
  MCLS_DomainTransporter_impl.hpp
  MCLS_Events.hpp
  MCLS_GlobalRNG.hpp
  MCLS_History.hpp
  MCLS_History_impl.hpp
  MCLS_HistoryBuffer.hpp
  MCLS_HistoryBuffer_impl.hpp
  MCLS_LinearProblem.hpp
  MCLS_LinearProblem_impl.hpp
  MCLS_MatrixTraits.hpp
  MCLS_RNGControl.hpp
  MCLS_SamplingTools.hpp
  MCLS_Serializer.hpp
  MCLS_Source.hpp
  MCLS_SourceTransporter.hpp
  MCLS_SourceTransporter_impl.hpp
  MCLS_SPRNG.hpp
  MCLS_UniformAdjointSource.hpp
  MCLS_UniformAdjointSource_impl.hpp
  MCLS_VectorExport.hpp
  MCLS_VectorTraits.hpp
  ) 

APPEND_SET(SOURCES
  MCLS_DBC.cpp
  MCLS_GlobalRNG.cpp
  MCLS_RNGControl.cpp
  MCLS_SPRNG.cpp
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
