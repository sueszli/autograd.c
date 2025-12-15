#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "unity::framework" for configuration ""
set_property(TARGET unity::framework APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(unity::framework PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libunity.a"
  )

list(APPEND _cmake_import_check_targets unity::framework )
list(APPEND _cmake_import_check_files_for_unity::framework "${_IMPORT_PREFIX}/lib/libunity.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
