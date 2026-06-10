set(REQUIRED_HEADERS json.hpp)
set(IMPORTED_TARGETS_FOR_ALL_LIBS nlohmann_json::nlohmann_json)

tribits_tpl_allow_pre_find_package(json json_ALLOW_PREFIND)

set(json_ALLOW_DOWNLOAD "OFF" CACHE BOOL "Allow CMake to download nlohmann/json")

if (json_ALLOW_PREFIND)
  message("-- Using find_package(nlohmann_json ...) ...")
  find_package(nlohmann_json)
  if (nlohmann_json_FOUND)
    # message("-- Found GTest_DIR='${GTest_DIR}'")
    # message("-- Generating gtest::all_libs and gtestConfig.cmake")
    tribits_extpkg_create_imported_all_libs_target_and_config_file(json
      INNER_FIND_PACKAGE_NAME nlohmann_json
      IMPORTED_TARGETS_FOR_ALL_LIBS  ${IMPORTED_TARGETS_FOR_ALL_LIBS})
  else()

    if (json_ALLOW_DOWNLOAD)
      message("-- Attempting to download nlohmann/json")
      include(FetchContent)
      FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        55f93686c01528224f448c19128836e7df245f72 # release 3.12.0
        FIND_PACKAGE_ARGS
      )
      FetchContent_MakeAvailable(json)
      find_package(json REQUIRED)

      add_library(json::all_libs INTERFACE IMPORTED GLOBAL)
      target_link_libraries(json::all_libs INTERFACE ${IMPORTED_TARGETS_FOR_ALL_LIBS})

    else ()

      message(WARNING "json was not found. Either install it or allow CMake to download it by setting json_ALLOW_DOWNLOAD:BOOL=ON")

    endif()
  endif()
endif()

if (NOT TARGET json::all_libs)

  TRIBITS_TPL_FIND_INCLUDE_DIRS_AND_LIBRARIES(json
    REQUIRED_HEADERS ${REQUIRED_HEADERS}
  )

endif()
