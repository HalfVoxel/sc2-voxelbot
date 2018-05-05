cmake_minimum_required(VERSION 3.1)

project(bot)

# Use bin as the directory for all executables.
# This will make protoc easy to find.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Windows builds subdirectories Debug/Release.
# These variables will overwrite that and put binaries in bin.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${PROJECT_BINARY_DIR}/bin)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${PROJECT_BINARY_DIR}/bin)

# Build with c++14 support.
set(CMAKE_CXX_STANDARD 14)

# Also compile the Starcraft 2 c++ API
add_subdirectory("s2client-api")

# Add include directories for the SC2 API (the examples one should be removed eventually when our bot doesn't depend on it)
include_directories("s2client-api/include")
include_directories("s2client-api/examples/common")

# Setup MSVC parallelized builds
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)

# More dependencies
include_directories(SYSTEM "${PROJECT_BINARY_DIR}/s2client-api/generated")
# Doesn't seem to be required, so it is commented out for now (might be required on Windows or something)
# include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/s2client-api/contrib/SDL-mirror/include")

# Function to generate an example project with extra libraries
function (example_project_extra project_name source_code extra_libs)
	# TODO: The .h files don't seem to be necessary (think only .cpp files should be included here anyway)
    file (GLOB example_project_sources ${source_code} "s2client-api/examples/common/*.h" "s2client-api/examples/common/*.cc" "bot/*.h")
    add_executable(${project_name} ${example_project_sources})

    # Sets the grouping in IDEs like visual studio (last parameter is the group name)
    set_target_properties(${project_name} PROPERTIES FOLDER target)

	if (MSVC)
		# set_target_properties(${project_name} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:libcmt;libconcrt")
	endif ()

	if (MSCV AND (NOT (MSVC_VERSION LESS 1910)))
		target_link_libraries(${project_name} legacy_stdio_definitions)
	endif ()

    target_link_libraries(${project_name} sc2api sc2lib sc2utils ${extra_libs})
endfunction ()

example_project_extra(our_bot "bot/*.cpp" "")