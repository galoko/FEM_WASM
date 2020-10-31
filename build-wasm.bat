call emsdk_env
em++ ^
-I include/Eigen/ ^
-s WASM=1 -s MODULARIZE=1 -O3 ^
-s "EXPORTED_FUNCTIONS=['_malloc', '_free', '_setupSimulation', '_tick']" ^
src/cpp/main.cpp ^
src/cpp/Physics.cpp ^
-o src/wasm/FEM.js
