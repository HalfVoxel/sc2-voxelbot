all:
	mkdir -p build
	cd build && cmake .. && make

run: build
	cd build && bin/our_bot
