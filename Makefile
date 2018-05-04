all: init build

init:
	cd s2client-api && ./cmake_gmake.sh

build:
	cd s2client-api/build_gmake && make

run: build
	s2client-api/build_gmake/bin/our_bot
