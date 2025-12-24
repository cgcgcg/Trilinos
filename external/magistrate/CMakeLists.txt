TRIBITS_PACKAGE(magistrate)
INCLUDE(FetchContent)


MACRO(apply_patch_to_magistrate MAGISTRATE_REPO_SOURCE_DIR)
  #revert the file for multiple reconfigures
  execute_process(
    COMMAND git checkout src/CMakeLists.txt
    WORKING_DIRECTORY ${MAGISTRATE_REPO_SOURCE_DIR}
    RESULT_VARIABLE GIT_APPLY_RESULT
    OUTPUT_VARIABLE GIT_APPLY_OUTPUT
    ERROR_VARIABLE GIT_APPLY_ERROR
  )

  set(PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/fix-export-targets-for-trilinos.patch")
  execute_process(
    COMMAND git apply ${PATCH_FILE}
    WORKING_DIRECTORY ${MAGISTRATE_REPO_SOURCE_DIR}  # Change to the appropriate directory if needed
    RESULT_VARIABLE GIT_APPLY_RESULT
    OUTPUT_VARIABLE GIT_APPLY_OUTPUT
    ERROR_VARIABLE GIT_APPLY_ERROR
  )
  if(NOT GIT_APPLY_RESULT EQUAL 0)
    message(FATAL_ERROR "git apply failed with error: ${GIT_APPLY_ERROR}\nOutput: ${GIT_APPLY_OUTPUT}")
  else()
    message(STATUS "Patching magistrate for Trilinos succeeded")
  endif()
ENDMACRO()

OPTION(${PACKAGE_NAME}_DOWNLOAD_SOURCE "If \"ON\" then magistrate source code will be downloaded from the internet." OFF)

IF(${PROJECT_NAME}_ENABLE_magistrate)
  IF(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/magistrate")
    MESSAGE(STATUS "Enabling inline source build of magistrate = ${HAVE_TEUCHOS_MAGISTRATE}")
    SET(MAGISTRATE_REPO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/magistrate")
    MESSAGE(STATUS "magistrate source dir = ${MAGISTRATE_REPO_SOURCE_DIR}")
    APPLY_PATCH_TO_MAGISTRATE(${MAGISTRATE_REPO_SOURCE_DIR})
    ADD_SUBDIRECTORY(magistrate)
  ELSE()
    MESSAGE(FATAL_ERROR "ERROR: magistrate was enabled but could not find magistrate source repository. Please check out the repo under the Trilinos/external/magistrate source directory.")
  ENDIF()

  ADD_LIBRARY(magistrate::all_libs ALIAS ${MAGISTRATE_LIBRARY})
ENDIF()

TRIBITS_PACKAGE_POSTPROCESS()
