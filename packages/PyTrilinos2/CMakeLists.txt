# -*- cmake -*-

# @HEADER
# ***********************************************************************
#
#          PyTrilinos2: Automatic Python Interfaces to Trilinos Packages
#                 Copyright (2022) Sandia Corporation
#
# Under the terms of Contract DE-AC04-94AL85000 with Sandia
# Corporation, the U.S. Government retains certain rights in this
# software.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the Corporation nor the names of the
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Questions? Contact Kim Liegeois (knliege@sandia.gov)
#
# ***********************************************************************
# @HEADER

# Set the package name
TRIBITS_PACKAGE(PyTrilinos2 DISABLE_STRONG_WARNINGS)

IF(NOT BUILD_SHARED_LIBS)
  MESSAGE(FATAL_ERROR "PyTrilinos2 can only be built with shared libraries. Building of shared libraries is currently set to OFF. To enable shared libraries please set the cache variable \"BUILD_SHARED_LIBS\" to ON")
ENDIF()

# Set the package version number
SET(PyTrilinos2_VERSION ${Trilinos_VERSION})

# Binder generation
IF (${PACKAGE_NAME}_ENABLE_BINDER)
  SET(PYTRILINOS2_GENERATE_SRC YES)
ELSE()
  SET(PYTRILINOS2_GENERATE_SRC NO)
ENDIF()

MESSAGE("-- PYTRILINOS2_PYTHON_EXECUTABLE:")
IF(NOT DEFINED ${PYTRILINOS2_PYTHON_EXECUTABLE})
  find_program(PYTRILINOS2_PYTHON_EXECUTABLE
      NAMES python3 python
      )
  MESSAGE("  -- CMake has set: PYTRILINOS2_PYTHON_EXECUTABLE = ${PYTRILINOS2_PYTHON_EXECUTABLE}")
ELSE()
  MESSAGE("  -- User has set: PYTRILINOS2_PYTHON_EXECUTABLE = ${PYTRILINOS2_PYTHON_EXECUTABLE}")
ENDIF()

IF (PYTRILINOS2_GENERATE_SRC)
  MESSAGE("-- PYTRILINOS2_BINDER_EXECUTABLE:")
  IF(NOT DEFINED PYTRILINOS2_BINDER_EXECUTABLE)
    find_program(PYTRILINOS2_BINDER_EXECUTABLE
        NAMES binder
        )
    MESSAGE("  -- CMake has set: PYTRILINOS2_BINDER_EXECUTABLE = ${PYTRILINOS2_BINDER_EXECUTABLE}")
  ELSE()
    MESSAGE("  -- User has set: PYTRILINOS2_BINDER_EXECUTABLE = ${PYTRILINOS2_BINDER_EXECUTABLE}")
  ENDIF()
ELSE()
  # Copy binder files to ${CMAKE_CURRENT_BINARY_DIR}
  FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/binder DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
ENDIF()

include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        v2.6.2
	GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(pybind11)

EXECUTE_PROCESS(COMMAND
  ${PYTHON_EXECUTABLE} -c "import mpi4py; print(mpi4py.get_include())"
  OUTPUT_VARIABLE Mpi4Py_INCLUDE_DIR
  ERROR_VARIABLE  Mpi4Py_INCLUDE_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

add_subdirectory( src )

# Execute the package postprocessing
TRIBITS_PACKAGE_POSTPROCESS()
