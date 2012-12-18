#
# See documentation in Trilinos preCopyrightTrilinos/ExtraExternalRepositories.cmake
#

INCLUDE(TribitsListHelpers)

SET( Chimera_PACKAGES_AND_DIRS_AND_CLASSIFICATIONS
  Chimera         .     SS
  )

PACKAGE_DISABLE_ON_PLATFORMS(Chimera Windows)
