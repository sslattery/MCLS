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

# Design-by-Contract
TRIBITS_ADD_OPTION_AND_DEFINE(
	${PACKAGE_NAME}_ENABLE_DBC
	HAVE_MCLS_DBC
	"Enable Design-by-Contract checks. WARNING: use for debug checking but disable for production runs as it incurs a significant overhead."
	${${PROJECT_NAME}_ENABLE_DEBUG}
)

TRIBITS_ADD_DEBUG_OPTION()

TRIBITS_ADD_SHOW_DEPRECATED_WARNINGS_OPTION()

##---------------------------------------------------------------------------##
## C) Add library, test, and examples.
##---------------------------------------------------------------------------##

ADD_SUBDIRECTORY(src)

TRIBITS_ADD_TEST_DIRECTORIES(test)

ADD_SUBDIRECTORY(tpetra)

TRIBITS_ADD_EXAMPLE_DIRECTORIES(example)

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Epetra)
ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_EpetraExt)
IF (${PACKAGE_NAME}_ENABLE_Epetra)
  IF (${PACKAGE_NAME}_ENABLE_EpetraExt)
    ADD_SUBDIRECTORY(epetra)
  ENDIF()
ENDIF()

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Thyra)
IF (MCLS_ENABLE_Thyra)
  ADD_SUBDIRECTORY(thyra)
ENDIF()

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Stratimikos)
IF (MCLS_ENABLE_Stratimikos)
  ADD_SUBDIRECTORY(stratimikos)
ENDIF()

##---------------------------------------------------------------------------##
## D) Do standard postprocessing
##---------------------------------------------------------------------------##

TRIBITS_PACKAGE_POSTPROCESS()

##---------------------------------------------------------------------------##
##                      end of MCLS/CMakeLists.txt
##---------------------------------------------------------------------------##
