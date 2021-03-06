ifeq ($(OS),Windows_NT)
	BUILD = cmake ../ -G "Visual Studio 15 2017 Win64"
else
	# BUILD = cmake -DCMAKE_BUILD_TYPE=Debug -DPYTHON_EXECUTABLE:FILEPATH="/Users/arong/anaconda3/bin/python" .. && make -j8
	BUILD = cmake -DCMAKE_BUILD_TYPE=RelWithDebug .. && make -j8
	# BUILD = cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j8
endif

all:
	mkdir -p build
	cd build && $(BUILD)

debug: FORCE
	mkdir -p debug
	cd debug && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j8

run: all
	cd build && bin/our_bot

composition: all
	cd build && bin/our_bot --composition

format:
	clang-format -i bot/*.cpp bot/*.h

FORCE: ;