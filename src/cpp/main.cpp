#include <emscripten.h>

#include "Physics.h"

extern "C" {
	void setupSimulation(float *vertices, int verticesCount, unsigned int *tetsIndices, int tetsIndicesCount, float wallsSize) {
		Physics::getInstance().initialize(vertices, verticesCount, tetsIndices, tetsIndicesCount, wallsSize);
	}

    void tick(double dt) {
		Physics::getInstance().advance(dt);
	}
}