ifeq ($(OS),Windows_NT)
	BUILD = cmake ../ -G "Visual Studio 15 2017 Win64"
else
	# BUILD = cmake -DCMAKE_BUILD_TYPE=Debug -DPYTHON_EXECUTABLE:FILEPATH="/Users/arong/anaconda3/bin/python" .. && make -j8
	BUILD = cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE:FILEPATH="/Users/arong/anaconda3/bin/python" .. && make -j8
endif

all:
	mkdir -p build
	cd build && $(BUILD)

run: all
	cd build && bin/our_bot

composition: all
	cd build && bin/our_bot --composition

format:
	clang-format -i bot/*.cpp bot/*.h