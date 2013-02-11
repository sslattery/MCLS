INCLUDE(TribitsLibraryMacros)

#
# A) Package-specific configuration options
#

#
# B) Define the header and source files (and directories)
#

SET(HEADERS "")
SET(SOURCES "")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

APPEND_SET(HEADERS
  MCLS_LinearProblemAdapter.hpp
  MCLS_MultiVectorTraits.hpp
  MCLS_SolverManagerAdapter.hpp
  Thyra_MCLSLinearOpWithSolve.hpp
  Thyra_MCLSLinearOpWithSolve_impl.hpp
  Thyra_MCLSLinearOpWithSolveFactory.hpp
  Thyra_MCLSLinearOpWithSolveFactory_impl.hpp
  ) 

APPEND_SET(SOURCES
  )

#
# C) Define the targets for package's library(s)
#

TRIBITS_ADD_LIBRARY(
  mclsthyra
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  DEPLIBS mcls thyracore thyraepetra thyratpetra
  )

INSTALL(FILES 
  ${HEADERS} 
  DESTINATION 
  ${CMAKE_INSTALL_PREFIX}/include/
)
