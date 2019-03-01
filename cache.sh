#!/bin/bash
make && ./build/bin/caching > bot/generated/abilities.txt
make && ./build/bin/caching pass2
