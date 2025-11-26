
INCLUDE(TrilinosCreateClientTemplateHeaders)
INCLUDE(ETI_functions)

#
# A) Package-specific configuration options
#

# TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_config.h)

#
# B) Define the header and source files (and directories)
#

#
# src
#

SET(HEADERS "")
SET(SOURCES "")

# SET(HEADERS ${HEADERS}
#   ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.h
#   )

#
# Core Files
#

# general headers, sources
TRIBITS_SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_SOURCE_DIR})
APPEND_GLOB(HEADERS ${DIR}/*.hpp ${DIR}/*.h)
APPEND_GLOB(SOURCES ${DIR}/*.cpp ${DIR}/*.c)
TRILINOS_CREATE_CLIENT_TEMPLATE_HEADERS(${DIR})

# Must glob the binary dir last to get all of the auto-generated headers
TRIBITS_SET_AND_INC_DIRS(DIR ${CMAKE_CURRENT_BINARY_DIR})
APPEND_GLOB(HEADERS ${DIR}/*.hpp)


IF (${PACKAGE_NAME}_ENABLE_EXPLICIT_INSTANTIATION)

  # Generate ETI .cpp files for Tpetra::MatrixMarketReaderWriter.
  TPETRA_PROCESS_ALL_SLGN_TEMPLATES(MatrixMarketReaderWriter_OUTPUT_FILES "Tpetra_ETI_SC_LO_GO_NT.tmpl" "IO_MatrixMarket" "IO_MATRIXMARKET" "${TpetraCore_ETI_SCALARS}" "${TpetraCore_ETI_LORDS}" "${TpetraCore_ETI_GORDS}" "${TpetraCore_ETI_NODES}" TRUE)
  LIST(APPEND SOURCES ${MatrixMarketReaderWriter_OUTPUT_FILES})

  # Generate ETI .cpp files for Tpetra::MatrixMarketReaderWriter.
  TPETRA_PROCESS_ALL_SLGN_TEMPLATES(BinaryReaderWriter_OUTPUT_FILES "Tpetra_ETI_SC_LO_GO_NT.tmpl" "IO_Binary" "IO_BINARY" "${TpetraCore_ETI_SCALARS}" "${TpetraCore_ETI_LORDS}" "${TpetraCore_ETI_GORDS}" "${TpetraCore_ETI_NODES}" TRUE)
  LIST(APPEND SOURCES ${BinaryReaderWriter_OUTPUT_FILES})

ENDIF()

#
# C) Define the targets for package's library(s)
#

TRIBITS_ADD_LIBRARY(
  tpetrainout
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  DEPLIBS tpetra
  )

TRIBITS_ADD_TEST_DIRECTORIES(examples)

#
# Make a trivial change to this comment if you add / remove a file
# either to / from this directory, or to / from the 'impl'
# subdirectory.  That ensures that running "make" will also rerun
# CMake in order to regenerate Makefiles.  Here is another such
# change.
#
