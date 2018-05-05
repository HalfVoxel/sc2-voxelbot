ifeq ($(OS),Windows_NT)
	BUILD = cmake ../ -G "Visual Studio 15 2017 Win64"
else
	BUILD = cmake .. && make
endif

all:
	mkdir -p build
	cd build && $(BUILD)

run: build
	cd build && bin/our_bot

format:
	clang-format -i bot/*.cpp