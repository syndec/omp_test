#! /bin/bash
# auto run
if [ ! -d "build" ]; then
    mkdir build
fi

cd build
rm -rf *

echo "run cmake and make."
cmake ..
make&&make install

