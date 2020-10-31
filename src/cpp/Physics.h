#pragma once

#include <string>

#include <vector>
#include "Eigen/Sparse"

#include "EigenTypes.h"
#include "WASM_math.h"

using namespace std;

class Physics {
public:
    static Physics& getInstance() {
        static Physics instance;

        return instance;
    }

    Physics(Physics const&) = delete;
    void operator=(Physics const&)  = delete;
private:
    Physics();

    int initialized;

    unsigned int nVerts;
    unsigned int nTets;
    unsigned int vecSize;

    EigenVector3 gravity;

    //Cholesky factorization of the system matrix
    Eigen::SparseMatrix<float, Eigen::ColMajor> matL;
    Eigen::SparseMatrix<float, Eigen::ColMajor> matLT;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> permInv;
    //temporal varialbes of the solver
    std::vector<EigenVector3> x_old;
    std::vector<Scalarf4, AlignmentAllocator<Scalarf4, 16>> RHS;
    std::vector<Scalarf4, AlignmentAllocator<Scalarf4, 16>> RHS_perm;
    std::vector<Scalarf4, AlignmentAllocator<Scalarf4, 16>> Kvec;
    std::vector<std::vector<std::vector<Scalarf4, AlignmentAllocator<Scalarf4, 16>>>> DT;
    std::vector<Quaternion4f, AlignmentAllocator<Quaternion4f, 16> > quats;
    //variables for the volume constraints
    std::vector<std::vector<int>> volume_constraint_phases;
    std::vector<std::vector<Scalarf4, AlignmentAllocator<Scalarf4, 16>>> rest_volume_phases;
    std::vector<std::vector<Scalarf4, AlignmentAllocator<Scalarf4, 16>>> alpha_phases;
    std::vector<std::vector<Scalarf4, AlignmentAllocator<Scalarf4, 16>>> kappa_phases;
    std::vector<std::vector<std::vector<Scalarf4, AlignmentAllocator<Scalarf4, 16>>>> inv_mass_phases;

    EigenVector3 wallsPosition;
    float wallsSize;

    float *vertices;
    unsigned int verticesCount;
    vector<EigenVector3> verticesVector;

    unsigned int *tetsIndices;
    unsigned int tetsIndicesCount;
    vector<vector<int>> tetsIndicesVector;

    const unsigned int HZ = 1000;
    const double dt = 1.0 / HZ;
    const unsigned int MAX_STEPS_LAG = HZ / 10;

    double timeAccumulator;

    void initializeModel();

    void convertToWASM(const vector<float>& v, vector<Scalarf4, AlignmentAllocator<Scalarf4, 16>>& vWASM);
    void convertToWASM(const vector<vector<vector<float>>>& v,
            vector<vector<vector<Scalarf4, AlignmentAllocator<Scalarf4, 16>>>>& vWASM);
    void initializeVolumeConstraints(const vector<vector<int>> &ind, vector<float> &rest_volume,
            vector<float> &invMass, float lambda);
    void constraintGraphColoring(const vector<vector<int>>& particleIndices, int n,
            vector<vector<int>>& coloring);

    void subStep();

    void processCollision(EigenVector3& position, EigenVector3& velocity);

    void solveOptimizationProblem(vector<EigenVector3> &p, const vector<vector<int>> &ind);
    inline void computeDeformationGradient(const vector<EigenVector3> &p, const vector<vector<int>> &ind,
            int i, Vector3f4 & F1, Vector3f4 & F2, Vector3f4 & F3);
    inline void APD_Newton_WASM(const Vector3f4& F1, const Vector3f4& F2, const Vector3f4& F3, Quaternion4f& q);

    void solveVolumeConstraints(vector<EigenVector3> &x, const vector<vector<int>> &ind);

    void benchmark();

    void transferLinearDataToVectors();
    void transferVectorsToLinearData();
public:
    void initialize(float *vertices, int verticesCount, unsigned int *tetsIndices, int tetsIndicesCount, float wallsSize);
    void finalize();

    void advance(double dt);

    EigenVector3& getGravity();
    void setGravity(EigenVector3& gravity);
};