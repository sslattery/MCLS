#
# See documentation in Trilinos preCopyrightTrilinos/ExtraExternalRepositories.cmake
#

INCLUDE(TribitsListHelpers)

SET( MCLS_PACKAGES_AND_DIRS_AND_CLASSIFICATIONS
  MCLS         .     SS
  )

TRIBITS_DISABLE_PACKAGE_ON_PLATFORMS(MCLS Windows)
