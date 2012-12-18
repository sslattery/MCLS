#
# See documentation in Trilinos preCopyrightTrilinos/ExtraExternalRepositories.cmake
#

INCLUDE(TribitsListHelpers)

SET( MCLS_PACKAGES_AND_DIRS_AND_CLASSIFICATIONS
  MCLS         .     SS
  )

PACKAGE_DISABLE_ON_PLATFORMS(MCLS Windows)
