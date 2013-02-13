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
  MCLS_EpetraAdapter.hpp
  MCLS_EpetraHelpers.hpp
  MCLS_EpetraRowMatrixAdapter.hpp
  MCLS_EpetraVectorAdapter.hpp
  MCLS_EpetraVectorExport.hpp
  ) 

APPEND_SET(SOURCES
  )

#
# C) Define the targets for package's library(s)
#

TRIBITS_ADD_LIBRARY(
  mclsepetra
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  DEPLIBS mcls epetra epetraext
  )

INSTALL(FILES 
  ${HEADERS} 
  DESTINATION 
  ${CMAKE_INSTALL_PREFIX}/include/
)
