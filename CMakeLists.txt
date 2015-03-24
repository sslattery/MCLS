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

TRIBITS_ADD_OPTION_AND_DEFINE(
        ${PACKAGE_NAME}_USE_ParaSails
	HAVE_MCLS_PARASAILS
	"Build MCLS with ParaSails preconditioners."
	${${PACKAGE_NAME}_ENABLE_ParaSails}
)

TRIBITS_ADD_OPTION_AND_DEFINE(
        ${PACKAGE_NAME}_USE_Temere
	HAVE_MCLS_TEMERE
	"Build MCLS with Temere preconditioners."
	${${PACKAGE_NAME}_ENABLE_Temere}
)

# Timings
TRIBITS_ADD_OPTION_AND_DEFINE(
	${PACKAGE_NAME}_ENABLE_TIMERS
	HAVE_MCLS_TIMERS
	"Enable timing."
	${${PROJECT_NAME}_ENABLE_DEBUG}
)

TRIBITS_ADD_DEBUG_OPTION()

TRIBITS_ADD_SHOW_DEPRECATED_WARNINGS_OPTION()

TRIBITS_ADD_ETI_INSTANTIATIONS(TpetraCore
  "S={double} LO={int} GO={int} N={${KokkosClassic_DefaultNode}}"
  "S={double} LO={int} GO={long} N={${KokkosClassic_DefaultNode}}"
  )

##---------------------------------------------------------------------------##
## C) Add library, test, and examples.
##---------------------------------------------------------------------------##

ADD_SUBDIRECTORY(src)

TRIBITS_ADD_TEST_DIRECTORIES(test)

ADD_SUBDIRECTORY(tpetra)

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Epetra)
ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_EpetraExt)
ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Ifpack)
ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_ParaSails)

IF (${PACKAGE_NAME}_ENABLE_Epetra)
  IF (${PACKAGE_NAME}_ENABLE_EpetraExt)
    IF (${PACKAGE_NAME}_ENABLE_Ifpack)
      IF (${PACKAGE_NAME}_ENABLE_ParaSails)
        ADD_SUBDIRECTORY(epetra)
      ENDIF()
    ENDIF()
  ENDIF()
ENDIF()

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Thyra)
IF (MCLS_ENABLE_Thyra)
  ADD_SUBDIRECTORY(thyra)
ENDIF()

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Stratimikos)
IF (MCLS_ENABLE_Thyra)
  IF (MCLS_ENABLE_Stratimikos)
    ADD_SUBDIRECTORY(stratimikos)
  ENDIF()
ENDIF()

TRIBITS_ADD_EXAMPLE_DIRECTORIES(example)

##---------------------------------------------------------------------------##
## D) Do standard postprocessing
##---------------------------------------------------------------------------##

TRIBITS_PACKAGE_POSTPROCESS()

##---------------------------------------------------------------------------##
##                      end of MCLS/CMakeLists.txt
##---------------------------------------------------------------------------##
