cmake_minimum_required(VERSION 3.1)

project(bot)

# Use bin as the directory for all executables.
# This will make protoc easy to find.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(PYTHON_EXECUTABLE:FILEPATH "/Users/arong/anaconda3/bin/python")

set(CMAKE_CONFIGURATION_TYPES, ${CMAKE_CONFIGURATION_TYPES} RelWithDebug)
set(CMAKE_CXX_FLAGS_RELWITHDEBUG "-O2 -g")

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
include_directories("cereal/include")

# Setup MSVC parallelized builds
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)

if(!MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fsdfsdfe")
endif()

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wmissing-field-initializers -Wuninitialized -D_GLIBCXX_DEBUG -fsanitize=address")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wmissing-field-initializers -Wuninitialized -O0 -fsanitize=address")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wmissing-field-initializers -Wuninitialized")

find_package(pybind11 REQUIRED)

# More dependencies
include_directories(SYSTEM "${PROJECT_BINARY_DIR}/s2client-api/generated")
# Doesn't seem to be required, so it is commented out for now (might be required on Windows or something)
# include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/s2client-api/contrib/SDL-mirror/include")

# Function to generate an example project with extra libraries
function (create_executable project_name source_code mainfile extra_libs)
	# TODO: The .h files don't seem to be necessary (think only .cpp files should be included here anyway)
    file (GLOB example_project_sources ${source_code} "s2client-api/examples/common/*.h" "s2client-api/examples/common/*.cc" "bot/*.h" "bot/behaviortree/*.h" "bot/utilities/*.h" "bot/generated/*.cpp" ${mainfile})
    add_executable(${project_name} ${example_project_sources})

    # Sets the grouping in IDEs like visual studio (last parameter is the group name)
    set_target_properties(${project_name} PROPERTIES FOLDER target)

	if (MSVC)
		# set_target_properties(${project_name} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:libcmt;libconcrt")
	endif ()

	if (MSCV AND (NOT (MSVC_VERSION LESS 1910)))
		target_link_libraries(${project_name} legacy_stdio_definitions)
	endif ()

    target_link_libraries(${project_name} sc2api sc2lib sc2utils pybind11::embed botlib ${extra_libs})
endfunction ()

function (create_library project_name source_code source_code2 source_code3 extra_libs)
	# TODO: The .h files don't seem to be necessary (think only .cpp files should be included here anyway)
    file (GLOB example_project_sources ${source_code} ${source_code2} ${source_code3} "s2client-api/examples/common/*.h" "s2client-api/examples/common/*.cc" "bot/*.h" "bot/behaviortree/*.h" "bot/utilities/*.h" "bot/generated/*.cpp" ${mainfile})
    add_library(${project_name} ${example_project_sources})

    if (MSCV AND (NOT (MSVC_VERSION LESS 1910)))
		target_link_libraries(${project_name} legacy_stdio_definitions)
	endif ()

    # Sets the grouping in IDEs like visual studio (last parameter is the group name)
    set_target_properties(${project_name} PROPERTIES FOLDER target)
    target_link_libraries(${project_name} sc2api sc2lib sc2utils pybind11::embed ${extra_libs})
endfunction ()

create_library(botlib "bot/*.cpp" "bot/behaviortree/*.cpp" "bot/utilities/*.cpp" "SDL2-static")
create_executable(our_bot "" "bot/bot/main.cpp" "SDL2-static")
create_executable(replay_analyzer "" "bot/replay_analyzer/main.cpp" "SDL2-static")
create_executable(caching "" "bot/caching/main.cpp" "SDL2-static")
create_executable(micro_train "" "bot/micro_train/micro_train.cpp" "SDL2-static")
create_executable(composition_analyzer "" "bot/composition_analyzer/main.cpp" "SDL2-static")
create_executable(composition_analyzer2 "" "bot/composition_analyzer2/main.cpp" "SDL2-static")
create_executable(build_optimizer "" "bot/build_optimizer/main.cpp" "SDL2-static")
create_executable(optimizer_train "" "bot/optimizer_Train/main.cpp" "SDL2-static")
create_executable(combat_predictor_train "" "bot/combat_predictor_train/main.cpp" "SDL2-static")

include_directories("s2client-api/contrib/SDL-mirror/include")
