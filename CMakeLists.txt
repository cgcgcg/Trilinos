#
# A) Define your project name and set up major project options
#

# To be safe, define your minimum CMake version.  This may be newer than the
# min required by TriBITS.
CMAKE_MINIMUM_REQUIRED(VERSION 3.27.0 FATAL_ERROR)

# Must set the project name as a variable at very beginning before including anything else
# We set the project name in a separate file so CTest scripts can use it.
INCLUDE(${CMAKE_CURRENT_LIST_DIR}/ProjectName.cmake)

# CMake requires that you declare the CMake project in the top-level file and
# not in an include file :-(
PROJECT(${PROJECT_NAME} NONE)

# Set up to use ccache
include("${CMAKE_CURRENT_LIST_DIR}/cmake/UseCCache.cmake")

# Set an env so we know we are in configure
set(ENV{CMAKE_IS_IN_CONFIGURE_MODE} 1)

# Don't define TriBITS override of include_directories()
set(TRIBITS_HIDE_DEPRECATED_INCLUDE_DIRECTORIES_OVERRIDE TRUE)

#
# B) Pull in the TriBITS system and execute
#

INCLUDE(${CMAKE_CURRENT_LIST_DIR}/cmake/tribits/TriBITS.cmake)

# Make Trilinos create <Package>Config.cmake files by default
SET(${PROJECT_NAME}_ENABLE_INSTALL_CMAKE_CONFIG_FILES_DEFAULT ON)
# Make Trilinos set up CPack support by default
SET(${PROJECT_NAME}_ENABLE_CPACK_PACKAGING_DEFAULT ON)
# Don't allow disabled subpackages to be excluded from tarball
SET(${PROJECT_NAME}_EXCLUDE_DISABLED_SUBPACKAGES_FROM_DISTRIBUTION_DEFAULT FALSE)


SET(Trilinos_USE_GNUINSTALLDIRS_DEFAULT ON)

SET(Trilinos_MUST_FIND_ALL_TPL_LIBS_DEFAULT TRUE)

FIND_PACKAGE(Doxygen)
IF(DOXYGEN_FOUND)
  SET(DOXYGEN_OUTPUT_DIR "" CACHE PATH "Path for doxygen documentation output")
  SET(DOXYGEN_TAG_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE PATH "Path for doxygen documentation output")

  ADD_SUBDIRECTORY(doc)
ENDIF()

GLOBAL_SET(DOCUMENTATION_TARGETS)

function(ADD_DOXYGEN_TARGET PKG_DIRNAME)
  ASSERT_DEFINED(DOXYGEN_FOUND)
  IF(DOXYGEN_FOUND)

    ASSERT_DEFINED(DOXYGEN_OUTPUT_DIR)

    IF ("${DOXYGEN_OUTPUT_DIR}" STREQUAL "")
      SET(${PACKAGE_NAME}_DOXYGEN_OUTPUT_DIR_DEFAULT "${CMAKE_CURRENT_BINARY_DIR}")
    ELSE ()
      SET(${PACKAGE_NAME}_DOXYGEN_OUTPUT_DIR_DEFAULT "${DOXYGEN_OUTPUT_DIR}/${PKG_DIRNAME}")
    ENDIF ()

    SET(${PACKAGE_NAME}_DOXYGEN_OUTPUT_DIR "${${PACKAGE_NAME}_DOXYGEN_OUTPUT_DIR_DEFAULT}" CACHE PATH "Path for ${PACKAGE_NAME} doxygen documentation output")

    SET(LOCAL_DOXYGEN_OUTPUT_DIR "${${PACKAGE_NAME}_DOXYGEN_OUTPUT_DIR}")

    SET(DOXYGEN_DEPENDENCY_TAGFILES "")
    FOREACH(dep in LISTS ${${PACKAGE_NAME}_FULL_ENABLED_DEP_PACKAGES})
      IF(TARGET doc_${dep})
        SET(DOXYGEN_DEPENDENCY_TAGFILES "${DOXYGEN_DEPENDENCY_TAGFILES} ${DOXYGEN_TAG_OUTPUT_DIR}/${dep}.tag=${${dep}_DOXYGEN_OUTPUT_DIR}/html")
      ENDIF()
    ENDFOREACH()

    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

    ADD_CUSTOM_TARGET(
      doc_${PACKAGE_NAME}
      ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating ${PACKAGE_NAME} API documentation with Doxygen"
      VERBATIM
    )
    # doxygen_add_docs(doc_${PACKAGE_NAME}
    #   CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    #   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    #   COMMENT "Generating ${PACKAGE_NAME} API documentation with Doxygen")
    APPEND_GLOBAL_SET(DOCUMENTATION_TARGETS "doc_${PACKAGE_NAME}")
  ENDIF(DOXYGEN_FOUND)

endfunction()

# Some CMake and TriBiTS tweaks just for Trilinos
include(TrilinosTweaks)

# Do all of the processing for this Tribits project
TRIBITS_PROJECT()

IF(DOXYGEN_FOUND)
  ADD_CUSTOM_TARGET(doc)
  FOREACH(target IN LISTS DOCUMENTATION_TARGETS)
    ADD_DEPENDENCIES(doc ${target})
  ENDFOREACH()
ENDIF()

INSTALL_BUILD_STATS_SCRIPTS()

# Install TriBITS so that other projects can use it
include(SetupTribitsInstall)

IF(${PROJECT_NAME}_ENABLE_YouCompleteMe)
  INCLUDE(CodeCompletion)
ENDIF()

message(STATUS "If publishing results using Trilinos, please cite us: https://trilinos.github.io/cite.html")
