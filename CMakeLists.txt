cmake_minimum_required(VERSION 3.1)

project(bot)

# Use bin as the directory for all executables.
# This will make protoc easy to find.
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(PYTHON_EXECUTABLE "/Users/arong/anaconda3/bin/python")

set(CMAKE_CONFIGURATION_TYPES, ${CMAKE_CONFIGURATION_TYPES} RelWithDebug)
set(CMAKE_CXX_FLAGS_RELWITHDEBUG "-O2 -g -fPIC")

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
add_subdirectory("pybind11")
add_subdirectory("libvoxelbot")

# Add include directories for the SC2 API (the examples one should be removed eventually when our bot doesn't depend on it)
include_directories("s2client-api/include")
include_directories("s2client-api/examples/common")
include_directories("cereal/include")

# Setup MSVC parallelized builds
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)

# if(!MSVC)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fsdfsdfe")
# endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wmissing-field-initializers -Wuninitialized -O0 -fno-omit-frame-pointer -DDISABLE_PYTHON=1 -fsanitize=address -DGLIBCXX_DEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wmissing-field-initializers -Wuninitialized -O0 -fno-omit-frame-pointer -DGLIBCXX_DEBUG")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -Wmissing-field-initializers -Wuninitialized -Wsometimes-uninitialized")
endif()

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wmissing-field-initializers -Wuninitialized -DDISABLE_PYTHON=1")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wmissing-field-initializers -Wuninitialized -D_GLIBCXX_DEBUG -fsanitize=address")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wmissing-field-initializers -Wuninitialized -Wsometimes-uninitialized")

# More dependencies
include_directories(SYSTEM "${PROJECT_BINARY_DIR}/s2client-api/generated")
# Doesn't seem to be required, so it is commented out for now (might be required on Windows or something)
# include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/s2client-api/contrib/SDL-mirror/include")

# Function to generate an example project with extra libraries
function (create_executable project_name source_code mainfile extra_libs)
	# TODO: The .h files don't seem to be necessary (think only .cpp files should be included here anyway)
    file (GLOB example_project_sources ${source_code} "bot/*.h" "bot/behaviortree/*.h" "bot/ml/*.h" "bot/utilities/*.h" "bot/generated/*.cpp" ${mainfile})
    add_executable(${project_name} ${example_project_sources})

    # Sets the grouping in IDEs like visual studio (last parameter is the group name)
    set_target_properties(${project_name} PROPERTIES FOLDER target)

	if (MSVC)
		# set_target_properties(${project_name} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:libcmt;libconcrt")
	endif ()

	if (MSCV AND (NOT (MSVC_VERSION LESS 1910)))
		target_link_libraries(${project_name} legacy_stdio_definitions)
	endif ()

    target_link_libraries(${project_name} sc2api sc2lib sc2utils pybind11::embed botlib libvoxelbot ${extra_libs})
endfunction ()

function (create_library project_name source_code extra_libs)
	# TODO: The .h files don't seem to be necessary (think only .cpp files should be included here anyway)
    # file (GLOB example_project_sources ${source_code} "s2client-api/examples/common/*.h" "s2client-api/examples/common/*.cc" "bot/*.h" "bot/behaviortree/*.h" "bot/ml/*.h" "bot/utilities/*.h" "bot/generated/*.cpp")
    add_library(${project_name} ${source_code})
    # message(STATUS ${source_code})

    if (MSCV AND (NOT (MSVC_VERSION LESS 1910)))
		target_link_libraries(${project_name} legacy_stdio_definitions)
	endif ()

    # Sets the grouping in IDEs like visual studio (last parameter is the group name)
    set_target_properties(${project_name} PROPERTIES FOLDER target)
    target_link_libraries(${project_name} sc2api sc2lib sc2utils pybind11 libvoxelbot ${extra_libs})
endfunction ()

# Note: trying to add SDL2-static as the extra_libs parameter here causes pybind11 modules to fail to link (for unknown reasons, maybe PIC related?)
file (GLOB botlib_sources "bot/*.cpp" "bot/behaviortree/*.cpp" "s2client-api/examples/common/*.h" "bot/utilities/*.cpp" "bot/ml/*.cpp" "bot/generated/*.cpp" "bot/mcts/*.cpp")
# message(STATUS "${botlib_sources}")
create_library(botlib "${botlib_sources}" "")

create_executable(our_bot "" "bot/bot/main.cpp" "SDL2-static")
create_executable(replay_analyzer "" "bot/replay_analyzer/main.cpp" "SDL2-static")
create_executable(map_exporter "" "bot/replay_analyzer/map_exporter.cpp" "SDL2-static")
create_executable(cache_unit_mappings "" "bot/caching/main_py.cpp" "SDL2-static")
create_executable(micro_train "" "bot/micro_train/micro_train.cpp" "SDL2-static")
create_executable(composition_analyzer "" "bot/composition_analyzer/main.cpp" "SDL2-static")
create_executable(composition_analyzer2 "" "bot/composition_analyzer2/main.cpp" "SDL2-static")
create_executable(build_optimizer "" "bot/build_optimizer/main.cpp" "SDL2-static")
create_executable(optimizer_train "" "bot/optimizer_train/main.cpp" "SDL2-static")
create_executable(build_order_train "" "bot/build_order_train/main.cpp" "SDL2-static")
create_executable(combat_predictor_train "" "bot/combat_predictor_train/main.cpp" "SDL2-static")
create_executable(mcts_test "" "bot/entrypoints/mcts_test.cpp" "SDL2-static")
# create_executable(test_simulator "" "bot/entrypoints/test_simulator.cpp" "SDL2-static")
create_executable(mcts_cache_unit_tests "" "bot/entrypoints/mcts_cache_unit_tests.cpp" "SDL2-static")
create_executable(view_replay "" "bot/entrypoints/view_replay.cpp" "SDL2-static")
create_executable(experiment_combat "" "bot/entrypoints/experiment_combat.cpp" "SDL2-static")
create_executable(experiment_buildorder "" "bot/entrypoints/experiment_buildorder.cpp" "SDL2-static")
create_executable(experiment_buildorder_sim "" "bot/entrypoints/experiment_buildorder_sim.cpp" "SDL2-static")
create_executable(debug_bo "" "bot/entrypoints/debug_bo.cpp" "SDL2-static")
create_executable(debug_army_comp "" "bot/entrypoints/debug_army_comp.cpp" "SDL2-static")
create_executable(experiment_compare_buildorder "" "bot/entrypoints/experiment_compare_buildorder.cpp" "SDL2-static")

file (GLOB build_order_train_sources "s2client-api/examples/common/*.h" "bot/*.h" "bot/behaviortree/*.h" "bot/utilities/*.h" "bot/ml/*.h" "bot/generated/*.cpp" "bot/build_order_train/main.cpp")
# pybind11_add_module(cppbot MODULE ${build_order_train_sources})
# set_target_properties(cppbot PROPERTIES CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
# target_link_libraries(cppbot PRIVATE pybind11::module sc2api sc2lib sc2utils botlib)

pybind11_add_module(testmod MODULE "bot/build_order_train/test.cpp")
# target_link_libraries(testmod PUBLIC pybind11::module sc2api sc2lib sc2utils botlib)
target_link_libraries(testmod PUBLIC pybind11::module sc2api sc2lib sc2utils botlib)
# get_target_property(OUT testmod LINK_LIBRARIES)
# message(STATUS ${OUT})

pybind11_add_module(botlib_bindings MODULE "bot/python_bindings/bindings.cpp")
target_link_libraries(botlib_bindings PUBLIC pybind11::module sc2api sc2lib sc2utils botlib SDL2-static)

include_directories("s2client-api/contrib/SDL-mirror/include")
