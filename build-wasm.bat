call emsdk_env
em++ ^
-I include/Eigen/ ^
-s WASM=1 -s MODULARIZE=1 -O3 ^
-s "EXPORTED_FUNCTIONS=['_malloc', '_free']" ^
src/cpp/main.cpp ^
-o src/wasm/FEM.js
