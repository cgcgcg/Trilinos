#ifndef TPETRA_SOURCELOCATION_HPP
#define TPETRA_SOURCELOCATION_HPP

#define INCLUDE_SOURCE_LOCATION_IN_KOKKOS_VIEW_LABEL 1

#if INCLUDE_SOURCE_LOCATION_IN_KOKKOS_VIEW_LABEL
# if __cplusplus >= 202002L
#  include <source_location>
#  define SOURCE_LOCATION_ARG , location
#  define SOURCE_LOCATION_DECL , const std::source_location location = std::source_location::source_location::current()
#  define SOURCE_LOCATION_DEF , const std::source_location location
#  define SOURCE_LOCATION_TRANSFORM(label) ViewLabel(label, location)
# else
#  include <experimental/source_location>
#  define SOURCE_LOCATION_ARG , location
#  define SOURCE_LOCATION_DECL , const std::experimental::source_location location = std::experimental::source_location::source_location::current()
#  define SOURCE_LOCATION_DEF , const std::experimental::source_location location
#  define SOURCE_LOCATION_TRANSFORM(label) Tpetra::ViewLabel(label, location)
# endif
#else
# define SOURCE_LOCATION_DECL
# define SOURCE_LOCATION_DEF
# define SOURCE_LOCATION_ARG
# define SOURCE_LOCATION_TRANSFORM(label) label
#endif

#if INCLUDE_SOURCE_LOCATION_IN_KOKKOS_VIEW_LABEL

#include <string>

namespace Tpetra {

  inline std::string ViewLabel(const std::string& label = ""
                               SOURCE_LOCATION_DECL) {
    return label + ((label.length() > 0) ? " " : "") +"(" + location.file_name() + ":" + std::to_string(location.line()) + ':' + std::to_string(location.column()) + " " + location.function_name() + ")";
  }
}
#endif

#endif
