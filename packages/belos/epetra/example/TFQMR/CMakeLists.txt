

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Triutils)
IF (${PACKAGE_NAME}_ENABLE_Triutils)
    
  ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Ifpack)
  IF (${PACKAGE_NAME}_ENABLE_Ifpack)

  TRIBITS_ADD_EXECUTABLE(
    TFQMR_Epetra_File_Ex
    SOURCES TFQMREpetraExFile.cpp 
    COMM serial mpi
  )

  TRIBITS_ADD_EXECUTABLE(
    Pseudo_Block_TFQMR_Epetra_File_Ex
    SOURCES PseudoBlockTFQMREpetraExFile.cpp 
    COMM serial mpi
  )

  TRIBITS_COPY_FILES_TO_BINARY_DIR(CopyExampleTFQMRFiles
    SOURCE_DIR ${Belos_SOURCE_DIR}/testmatrices
    SOURCE_FILES orsirr1.hb 
    EXEDEPS TFQMR_Epetra_File_Ex Pseudo_Block_TFQMR_Epetra_File_Ex
    )

  ENDIF(${PACKAGE_NAME}_ENABLE_Ifpack)

ENDIF(${PACKAGE_NAME}_ENABLE_Triutils)
