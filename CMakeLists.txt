##---------------------------------------------------------------------------##
## MCLS/CMakeLists.txt
## Stuart R. Slattery
##---------------------------------------------------------------------------##
## TriBITS package file for the MCLS package
##---------------------------------------------------------------------------##

INCLUDE(TribitsPackageMacros)
INCLUDE(TribitsAddOptionAndDefine)

##---------------------------------------------------------------------------##
## A) Define the package
##---------------------------------------------------------------------------##

TRIBITS_PACKAGE(MCLS)

##---------------------------------------------------------------------------##
## B) Set up package-specific options
##---------------------------------------------------------------------------##

TRIBITS_ADD_DEBUG_OPTION()

TRIBITS_ADD_SHOW_DEPRECATED_WARNINGS_OPTION()

##---------------------------------------------------------------------------##
## C) Add library, test, and examples.
##---------------------------------------------------------------------------##

ADD_SUBDIRECTORY(src)

TRIBITS_ADD_TEST_DIRECTORIES(test)

TRIBITS_ADD_EXAMPLE_DIRECTORIES(example)

IF (MCLS_ENABLE_Epetra)
  ADD_SUBDIRECTORY(epetra)
ENDIF()

IF (MCLS_ENABLE_Tpetra)
  ADD_SUBDIRECTORY(tpetra)
ENDIF()

##---------------------------------------------------------------------------##
## D) Do standard postprocessing
##---------------------------------------------------------------------------##

TRIBITS_PACKAGE_POSTPROCESS()

##---------------------------------------------------------------------------##
##                      end of MCLS/CMakeLists.txt
##---------------------------------------------------------------------------##
