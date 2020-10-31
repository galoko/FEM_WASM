extern "C" {
#include <time.h>
}

#include "Physics.h"

using namespace Eigen;

Physics::Physics() {

}

void Physics::initialize(float *vertices, int verticesCount, unsigned int *tetsIndices, int tetsIndicesCount, float wallsSize) {

    if (this->initialized == 1)
        return;

    gravity = EigenVector3(0, 0, -1) * 9.8f;
    gravity.normalize();

    wallsPosition = EigenVector3(0, 0, 0);
    this->wallsSize = wallsSize;

    this->vertices = vertices;
    this->verticesCount = verticesCount;
    this->tetsIndices = tetsIndices;
    this->tetsIndicesCount = tetsIndicesCount;

    this->initializeModel();

    this->timeAccumulator = 0;

    this->initialized = 1;

    // benchmark();
}

#define BILLION 1E9

double getTime(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec + time.tv_nsec / BILLION;
}

void Physics::benchmark() {

    const int STEPS_COUNT = 15000;

    double startTime = getTime();

    for (int i = 0; i < STEPS_COUNT; i++)
        subStep();

    double endTime = getTime();

    double elapsed = endTime - startTime;
    double elapsedSimulated = STEPS_COUNT * dt;

    double ratio = elapsedSimulated / elapsed;

    printf("Ratio: %f\n", ratio);
}

void Physics::transferLinearDataToVectors() {
    this->verticesVector = vector<EigenVector3>(this->verticesCount / 3);
    for (int i = 0; i < this->verticesVector.size(); i++) {
        this->verticesVector[i] = EigenVector3(
            this->vertices[i * 3 + 0], 
            this->vertices[i * 3 + 1], 
            this->vertices[i * 3 + 2]
        );
    }
    this->tetsIndicesVector = vector<vector<int>>(this->tetsIndicesCount / 4);
    for (int i = 0; i < this->tetsIndicesVector.size(); i++) {
        vector<int> indices = vector<int>(4);
        indices[0] = this->tetsIndices[i * 4 + 0];
        indices[1] = this->tetsIndices[i * 4 + 1];
        indices[2] = this->tetsIndices[i * 4 + 2];
        indices[3] = this->tetsIndices[i * 4 + 3];
        this->tetsIndicesVector[i] = indices;
    }
};

void Physics::transferVectorsToLinearData() {
    for (int i = 0; i < this->verticesVector.size(); i++) {
        EigenVector3 v = this->verticesVector[i];
        this->vertices[i * 3 + 0] = v.x();
        this->vertices[i * 3 + 1] = v.y();
        this->vertices[i * 3 + 2] = v.z();
    }
};

void Physics::initializeModel() {
    this->transferLinearDataToVectors();

    vector<EigenVector3>& p = this->verticesVector;
    vector<vector<int>>& ind = this->tetsIndicesVector;

    // bread
    float density = 190.0;
    float E =  0.3 * 1.0e6;
    float nu = 0.78;

    /*
    // copper
    float density = 2810;
    float E =  10 * 1.0e6;
    float nu = 0.32;
     */

    float mu = E / (2.0 * (1.0 + nu));
    float lambda = (E*nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));

    nVerts = this->verticesVector.size();
    nTets = this->tetsIndicesVector.size();
    if (nTets % 4 == 0) vecSize = nTets / 4;
    else vecSize = nTets / 4 + 1;

    vector<Triplet<float>> triplets_D;
    triplets_D.reserve(9 * nTets * 4);
    vector<Triplet<float>> triplets_K;
    triplets_K.reserve(9 * nTets);
    vector<Triplet<float>> triplets_M;
    triplets_M.reserve(4 * nTets);
    vector<float> Kreal(nTets);
    vector<vector<vector<float>>> Dt(nTets);
    vector<EigenMatrix3> Dm_inv(nTets);
    vector<float> rest_volume(nTets);
    vector<float> invMass(nVerts);

    //Algorithm 1, lines 1-12
    for (int t = 0; t < nTets; t++)
    {
        //indices of the 4 vertices of tet t
        vector<int>& it = ind[t];

        //compute rest pose shape matrix and volume
        EigenMatrix3 Dm;
        Dm.col(0) = p[it[1]] - p[it[0]];
        Dm.col(1) = p[it[2]] - p[it[0]];
        Dm.col(2) = p[it[3]] - p[it[0]];

        rest_volume[t] = 1.0 / 6.0 * Dm.determinant();
        assert(rest_volume[t] >= 0.0);

        Dm_inv[t] = Dm.inverse();

        //set triplets for the matrix K. Directly multiply the factor 2*dt*dt into K
        Kreal[t] = 2.0 * dt * dt * mu * rest_volume[t];

        for (int j = 0; j < 9; j++)
            triplets_K.push_back(Triplet<float>(9 * t + j, 9 * t + j, Kreal[t]));

        //initialize the lumped mass matrix
        for (int j = 0; j < 4; j++)		//forall verts of tet i
        {
            invMass[it[j]] += 0.25 * density * rest_volume[t];
            triplets_M.push_back(Triplet<float>(it[j], it[j], invMass[it[j]]));
        }

        //compute matrix D_t from Eq. (9) (actually Dt[t] is D_t^T)
        Dt[t].resize(4);

        Dt[t][0].resize(3);
        for (int k = 0; k < 3; k++)
            Dt[t][0][k] = -Dm_inv[t](0, k) - Dm_inv[t](1, k) - Dm_inv[t](2, k);

        for (int j = 1; j < 4; j++) {
            Dt[t][j].resize(3);
            for (int k = 0; k < 3; k++)
                Dt[t][j][k] = Dm_inv[t](j - 1, k);
        }

        //initialize the matrix D
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                triplets_D.push_back(Triplet<float>(9 * t + 3 * j, it[i], Dt[t][i][j]));
    }

    //set matrices
    SparseMatrix<float> K(9 * nTets, 9 * nTets);	// actually 2 * dt* dt * K
    SparseMatrix<float> D(9 * nTets, nVerts);
    SparseMatrix<float> M(nVerts, nVerts);
    K.setFromTriplets(triplets_K.begin(), triplets_K.end());
    D.setFromTriplets(triplets_D.begin(), triplets_D.end());
    M.setFromTriplets(triplets_M.begin(), triplets_M.end());

    //compute system matrix and Cholesky factorization (Algorithm 1, line 13)
    //remove the upper-left 3*nFixedVertices x 3*nFixedVertices block
    SparseMatrix<float> M_plus_DT_K_D = (M + D.transpose() * K * D).block(0, 0, nVerts, nVerts);

    SimplicialLLT<SparseMatrix<float>, Lower, AMDOrdering<int>> LLT;
    LLT.compute(M_plus_DT_K_D);
    perm = LLT.permutationP();
    permInv = LLT.permutationPinv();
    matL = SparseMatrix<float, ColMajor>(LLT.matrixL().cast<float>());
    matLT = SparseMatrix<float, ColMajor>(LLT.matrixU().cast<float>());

    //move data to vector registers
    Kvec.resize(vecSize);
    convertToWASM(Kreal, Kvec);
    DT.resize(vecSize);
    convertToWASM(Dt, DT);
    //prepare solver variables
    x_old.resize(nVerts);
    x_old.assign(p.begin(), p.end());
    quats.resize(vecSize);
    for (int i = 0; i < vecSize; i++)
        quats[i] = Quaternion4f(0, 0, 0, 1);
    RHS.resize(nVerts);
    RHS_perm.resize(nVerts);

    //initialize volume constraints
    for (size_t i = 0; i < invMass.size(); i++)
        invMass[i] = 1.0 / invMass[i];

    initializeVolumeConstraints(ind, rest_volume, invMass, lambda);
}

//initializes the volume constraints. For parallel Gauss-Seidel they are grouped with graph coloring
//the inverse masses, alpha values and rest volumes are moved to vector registers
void Physics::initializeVolumeConstraints(const vector<vector<int>> &ind, vector<float> &rest_volume,
        vector<float> &invMass, float lambda)
{
    constraintGraphColoring(ind, nVerts, volume_constraint_phases);

    inv_mass_phases.resize(volume_constraint_phases.size());
    rest_volume_phases.resize(volume_constraint_phases.size());
    kappa_phases.resize(volume_constraint_phases.size());
    alpha_phases.resize(volume_constraint_phases.size());

    for (int phase = 0; phase < volume_constraint_phases.size(); phase++)	//forall constraint phases
    {
        inv_mass_phases[phase].resize(0);
        rest_volume_phases[phase].resize(0);
        kappa_phases[phase].resize(0);
        alpha_phases[phase].resize(0);
        for (int c = 0; c<volume_constraint_phases[phase].size(); c += 4)	//forall constraints in phase
        {
            int c4[4];
            for (int k = 0; k < 4; k++)
                if (c + k < volume_constraint_phases[phase].size())
                    c4[k] = volume_constraint_phases[phase][c + k];

            float w0[4], w1[4], w2[4], w3[4], vol[4], alpha[4];
            for (int k = 0; k < 4; k++)
                if (c + k < volume_constraint_phases[phase].size())
                {
                    w0[k] = (float)invMass[ind[c4[k]][0]];
                    w1[k] = (float)invMass[ind[c4[k]][1]];
                    w2[k] = (float)invMass[ind[c4[k]][2]];
                    w3[k] = (float)invMass[ind[c4[k]][3]];

                    vol[k] = (float)rest_volume[c4[k]];
                    alpha[k] = 1.0f / (float)(lambda * rest_volume[c4[k]] * dt * dt);
                }
                else
                {
                    vol[k] = 1.0f;
                    alpha[k] = 0.0f;
                    w0[k] = (float)invMass[ind[c4[k]][0]];
                    w1[k] = (float)invMass[ind[c4[k]][1]];
                    w2[k] = (float)invMass[ind[c4[k]][2]];
                    w3[k] = (float)invMass[ind[c4[k]][3]];
                }

            int pos = (int)inv_mass_phases[phase].size();
            inv_mass_phases[phase].push_back(vector<Scalarf4, AlignmentAllocator<Scalarf4, 16>>(4));

            inv_mass_phases[phase][pos][0].load(w0);
            inv_mass_phases[phase][pos][1].load(w1);
            inv_mass_phases[phase][pos][2].load(w2);
            inv_mass_phases[phase][pos][3].load(w3);

            Scalarf4 restVol, alpha4;
            restVol.load(vol);
            alpha4.load(alpha);
            rest_volume_phases[phase].push_back(restVol);
            kappa_phases[phase].push_back(Scalarf4(0.0f));
            alpha_phases[phase].push_back(alpha4);
        }
    }
}

// this method is taken from the PBD library: https://github.com/InteractiveComputerGraphics/PositionBasedDynamics
void Physics::constraintGraphColoring(const vector<vector<int>>& particleIndices, int n,
        vector<vector<int>>& coloring)
{
    //particleIndices [constraint][particleIndex]
    vector<vector<bool>> particleColors;   //numColors x numParticles, true if particle in color
    particleColors.resize(0);
    coloring.resize(0);

    for (unsigned int i = 0; i < particleIndices.size(); i++)   //forall constraints
    {
        bool newColor = true;
        for (unsigned int j = 0; j < coloring.size(); j++)  //forall colors
        {
            bool addToThisColor = true;

            for (unsigned int k = 0; k < particleIndices[i].size(); k++) { //forall particles innvolved in the constraint
                if (particleColors[j][particleIndices[i][k]] == true) {
                    addToThisColor = false;
                    break;
                }
            }
            if (addToThisColor) {
                coloring[j].push_back(i);

                for (unsigned int k = 0; k < particleIndices[i].size(); k++) //forall particles innvolved in the constraint
                    particleColors[j][particleIndices[i][k]] = true;

                newColor = false;
                break;
            }
        }
        if (newColor) {
            particleColors.push_back(vector<bool>(n, false));
            coloring.resize(coloring.size() + 1);
            coloring[coloring.size() - 1].push_back(i);
            for (unsigned int k = 0; k < particleIndices[i].size(); k++) //forall particles innvolved in the constraint
                particleColors[coloring.size() - 1][particleIndices[i][k]] = true;
        }
    }
}

void Physics::convertToWASM(const vector<float>& v, vector<Scalarf4, AlignmentAllocator<Scalarf4, 16>>& vWASM)
{
    int regularPart = (nTets / 4) * 4;
    for (int i = 0; i < regularPart; i += 4)
        vWASM[i / 4] = Scalarf4(v[i + 0], v[i + 1], v[i + 2], v[i + 3]);

    if (regularPart != nTets)	//add padding with last value of v. (they are never read out)
        for (int j = 0; j < 3; j++)
        {
            float vtmp[4];
            for (int i = regularPart; i < regularPart + 4; i++)
                if (i < nTets) vtmp[i - regularPart] = v[i];
                else vtmp[i - regularPart] = v[nTets - 1];

            vWASM[regularPart / 4] = Scalarf4(vtmp[0], vtmp[1], vtmp[2], vtmp[3]);
        }
}

void Physics::convertToWASM(const vector<vector<vector<float>>>& v,
        vector<vector<vector<Scalarf4, AlignmentAllocator<Scalarf4, 16>>>>& vWASM)
{
    int regularPart = (nTets / 4) * 4;
    for (int i = 0; i < regularPart; i += 4)
    {
        vWASM[i / 4].resize(4);
        for (int j = 0; j < 4; j++)
        {
            vWASM[i / 4][j].resize(3);
            for (int k = 0; k < 3; k++)
                vWASM[i / 4][j][k] = Scalarf4(v[i + 0][j][k], v[i + 1][j][k], v[i + 2][j][k], v[i + 3][j][k]);
        }
    }

    if (regularPart != nTets) {	//add padding with last value of v. (they are never read out)
        vWASM[regularPart / 4].resize(4);
        for (int j = 0; j < 4; j++)
        {
            vWASM[regularPart / 4][j].resize(3);
            for (int k = 0; k < 3; k++)
            {
                float vtmp[4];
                for (int i = regularPart; i < regularPart + 4; i++)
                    if (i < nTets) vtmp[i - regularPart] = v[i][j][k];
                    else vtmp[i - regularPart] = v[nTets - 1][j][k];

                vWASM[regularPart / 4][j][k] = Scalarf4(vtmp[0], vtmp[1], vtmp[2], vtmp[3]);
            }
        }
    }
}

void Physics::finalize() {

    if (this->initialized == 0)
        return;

    x_old.clear();
    RHS.clear();
    RHS_perm.clear();
    Kvec.clear();
    DT.clear();
    quats.clear();
    volume_constraint_phases.clear();
    rest_volume_phases.clear();
    alpha_phases.clear();
    kappa_phases.clear();
    inv_mass_phases.clear();

    this->initialized = 0;
}

// thread

void Physics::advance(double dt) {
    this->timeAccumulator += dt;
    if (this->timeAccumulator > this->dt * this->MAX_STEPS_LAG) {
        this->timeAccumulator = this->dt * this->MAX_STEPS_LAG;
    }

    while (this->timeAccumulator >= this->dt) {
        this->subStep();
        this->timeAccumulator -= this->dt;
    }
    
    this->transferVectorsToLinearData();
}

// iterations

void Physics::subStep() {

    vector<EigenVector3>& x = this->verticesVector;
    vector<vector<int>>& ind = this->tetsIndicesVector;

    //explicit Euler to compute \tilde{x}
    for (int i = 0; i < nVerts; i++)
    {
        EigenVector3 v = (x[i] - x_old[i]) / dt;

        v += (dt * gravity);	//gravity

        processCollision(x[i], v);

        x_old[i] = x[i];
        x[i] += dt * v;
    }

    solveOptimizationProblem(x, ind);

    //solve volume constraints
    for (size_t i = 0; i < kappa_phases.size(); i++)	//reset Lagrange multipliers
        for (size_t j = 0; j < kappa_phases[i].size(); j++)
            kappa_phases[i][j] = Scalarf4(0.0f);

    for (int it = 0; it < 2; it++)	//solve constraints
        solveVolumeConstraints(x, ind);
}

void Physics::processCollision(EigenVector3& position, EigenVector3& velocity) {

    EigenVector3 low = wallsPosition - EigenVector3(wallsSize, wallsSize, wallsSize) * 0.5f;
    EigenVector3 high = wallsPosition + EigenVector3(wallsSize, wallsSize, wallsSize) * 0.5f;

    EigenVector3 error = EigenVector3(0, 0, 0);

    EigenVector3 delta;

    delta = position - low;
    error.x() = std::min(delta.x(), error.x());
    error.y() = std::min(delta.y(), error.y());
    error.z() = std::min(delta.z(), error.z());

    delta = position - high;
    error.x() = std::max(delta.x(), error.x());
    error.y() = std::max(delta.y(), error.y());
    error.z() = std::max(delta.z(), error.z());

    const float epsilon = 10e-7;

    if (fabs(error.x()) < epsilon && fabs(error.y()) < epsilon && fabs(error.z()) < epsilon)
        return;

    EigenVector3 normal = -error;
    normal.normalize();

    EigenVector3 normalVelocity = normal * velocity.dot(normal);

    EigenVector3 tangentVelocity = velocity - normalVelocity;

    const float friction = 1.0;

    tangentVelocity -= tangentVelocity * friction;

    normalVelocity = 500.0f * dt * normal;

    velocity = tangentVelocity + normalVelocity;
}

void Physics::solveOptimizationProblem(vector<EigenVector3> &p, const vector<vector<int>> &ind)
{
    //compute RHS of Equation (12)
    for (size_t i = 0; i < RHS.size(); i++)
        RHS[i] = Scalarf4(0.0f);

    for (int i = 0; i < vecSize; i++)
    {
        Vector3f4 F1, F2, F3;	//columns of the deformation gradient
        computeDeformationGradient(p, ind, i, F1, F2, F3);

        Quaternion4f& q = quats[i];
        APD_Newton_WASM(F1, F2, F3, q);

        //transform quaternion to rotation matrix
        Vector3f4 R1, R2, R3;	//columns of the rotation matrix
        quats[i].toRotationMatrix(R1, R2, R3);

        // R <- R - F
        R1 -= F1;
        R2 -= F2;
        R3 -= F3;

        //multiply with 2 * dt * dt * DT * K from left
        Vector3f4 dx[4];
        dx[0] = (R1 * DT[i][0][0] + R2 * DT[i][0][1] + R3 * DT[i][0][2]) * Kvec[i];
        dx[1] = (R1 * DT[i][1][0] + R2 * DT[i][1][1] + R3 * DT[i][1][2]) * Kvec[i];
        dx[2] = (R1 * DT[i][2][0] + R2 * DT[i][2][1] + R3 * DT[i][2][2]) * Kvec[i];
        dx[3] = (R1 * DT[i][3][0] + R2 * DT[i][3][1] + R3 * DT[i][3][2]) * Kvec[i];

        //write results to the corresponding positions in the RHS vector
        for(int k = 0; k < 4; k++)
        {
            float x[4], y[4], z[4];
            dx[k].x().store(x);
            dx[k].y().store(y);
            dx[k].z().store(z);

            for (int j = 0; j < 4; j++)
            {
                if(4 * i + j >= nTets) break;
                int pi = ind[4 * i + j][k];
                RHS[pi] += Scalarf4(x[j], y[j], z[j], 0.0);	//only first 3 comps are used, maybe use 128 bit registers
            }
        }
    }

    //solve the linear system
    //permutation of the RHS because of Eigen's fill-in reduction
    for (size_t i = 0; i < RHS.size(); i++)
        RHS_perm[perm.indices()[i]] = RHS[i];

    //foreward substitution
    for (int k = 0; k<matL.outerSize(); ++k)
        for (SparseMatrix<float, ColMajor>::InnerIterator it(matL, k); it; ++it)
            if (it.row() == it.col())
                RHS_perm[it.row()] = RHS_perm[it.row()] * Scalarf4(1.0 / it.value());
            else
                RHS_perm[it.row()] -= Scalarf4(it.value()) * RHS_perm[it.col()];

    //backward substitution
    for (int k = matLT.outerSize() - 1; k >= 0 ; --k)
        for (SparseMatrix<float, ColMajor>::ReverseInnerIterator it(matLT, k); it; --it)
            if (it.row() == it.col())
                RHS_perm[it.row()] = RHS_perm[it.row()] * Scalarf4(1.0 / it.value());
            else
                RHS_perm[it.row()] -= Scalarf4(it.value()) * RHS_perm[it.col()];

    //invert permutation
    for (size_t i = 0; i < RHS.size(); i++)
        RHS[permInv.indices()[i]] = RHS_perm[i];

    for (size_t i = 0; i<RHS.size(); i++)	// add result (delta_x) to the positions
    {
        float x[4];
        RHS[i].store(x);
        p[i].x() += x[0];
        p[i].y() += x[1];
        p[i].z() += x[2];
    }
}

//computes the deformation gradient of 8 tets
inline void Physics::computeDeformationGradient(const vector<EigenVector3> &p,
        const vector<vector<int>> &ind, int i, Vector3f4 & F1, Vector3f4 & F2, Vector3f4 & F3)
{
    Vector3f4 vertices[4];	//vertices of 8 tets
    int regularPart = (nTets / 4) * 4;
    int i4 = 4 * i;

    if (i4 < regularPart)
    {
        for (int j = 0; j < 4; j++)
        {
            const EigenVector3& p0_0 = p[ind[i4 + 0][j]];
            const EigenVector3& p0_1 = p[ind[i4 + 1][j]];
            const EigenVector3& p0_2 = p[ind[i4 + 2][j]];
            const EigenVector3& p0_3 = p[ind[i4 + 3][j]];

            vertices[j].x() = Scalarf4(p0_0[0], p0_1[0], p0_2[0], p0_3[0]);
            vertices[j].y() = Scalarf4(p0_0[1], p0_1[1], p0_2[1], p0_3[1]);
            vertices[j].z() = Scalarf4(p0_0[2], p0_1[2], p0_2[2], p0_3[2]);
        }
    }
    else    //add padding with vertices of last tet. (they are never read out)
    {
        for (int j = 0; j < 4; j++)
        {
            Vector3f p0[4];
            for (int k = regularPart; k < regularPart + 4; k++)
                if (k < nTets) p0[k - regularPart] = p[ind[k][j]].cast<float>();
                else p0[k - regularPart] = p[ind[nTets - 1][j]].cast<float>();

            vertices[j].x() = Scalarf4(p0[0][0], p0[1][0], p0[2][0], p0[3][0]);
            vertices[j].y() = Scalarf4(p0[0][1], p0[1][1], p0[2][1], p0[3][1]);
            vertices[j].z() = Scalarf4(p0[0][2], p0[1][2], p0[2][2], p0[3][2]);
        }
    }

    // compute F as D_t*x (see Equation (9))
    F1 = vertices[0] * DT[i][0][0] + vertices[1] * DT[i][1][0] + vertices[2] * DT[i][2][0] + vertices[3] * DT[i][3][0];
    F2 = vertices[0] * DT[i][0][1] + vertices[1] * DT[i][1][1] + vertices[2] * DT[i][2][1] + vertices[3] * DT[i][3][1];
    F3 = vertices[0] * DT[i][0][2] + vertices[1] * DT[i][1][2] + vertices[2] * DT[i][2][2] + vertices[3] * DT[i][3][2];
}

//computes the APD of 4 deformation gradients. (Alg. 3 from the paper)
inline void Physics::APD_Newton_WASM(const Vector3f4& F1, const Vector3f4& F2, const Vector3f4& F3, Quaternion4f& q)
{
    //one iteration is sufficient for plausible results
    for (int it = 0; it<1; it++)
    {
        //transform quaternion to rotation matrix
        Matrix3f4 R;
        q.toRotationMatrix(R);

        //columns of B = RT * F
        Vector3f4 B0 = R.transpose() * F1;
        Vector3f4 B1 = R.transpose() * F2;
        Vector3f4 B2 = R.transpose() * F3;

        Vector3f4 gradient(B2[1] - B1[2], B0[2] - B2[0], B1[0] - B0[1]);

        //compute Hessian, use the fact that it is symmetric
        Scalarf4 h00 = B1[1] + B2[2];
        Scalarf4 h11 = B0[0] + B2[2];
        Scalarf4 h22 = B0[0] + B1[1];
        Scalarf4 h01 = Scalarf4(-0.5) * (B1[0] + B0[1]);
        Scalarf4 h02 = Scalarf4(-0.5) * (B2[0] + B0[2]);
        Scalarf4 h12 = Scalarf4(-0.5) * (B2[1] + B1[2]);

        Scalarf4 detH = Scalarf4(-1.0) * h02 * h02 * h11 + Scalarf4(2.0) * h01 * h02 * h12 - h00 * h12 * h12 - h01 * h01 * h22 + h00 * h11 * h22;

        Vector3f4 omega;
        //compute symmetric inverse
        const Scalarf4 factor = Scalarf4(-0.25) / detH;
        omega[0] = (h11 * h22 - h12 * h12) * gradient[0]
                   + (h02 * h12 - h01 * h22) * gradient[1]
                   + (h01 * h12 - h02 * h11) * gradient[2];
        omega[0] *= factor;

        omega[1] = (h02 * h12 - h01 * h22) * gradient[0]
                   + (h00 * h22 - h02 * h02) * gradient[1]
                   + (h01 * h02 - h00 * h12) * gradient[2];
        omega[1] *= factor;

        omega[2] = (h01 * h12 - h02 * h11) * gradient[0]
                   + (h01 * h02 - h00 * h12) * gradient[1]
                   + (h00 * h11 - h01 * h01) * gradient[2];
        omega[2] *= factor;

        omega = Vector3f4::blend(abs(detH) < 1.0e-9f, gradient * Scalarf4(-1.0), omega);	//if det(H) = 0 use gradient descent, never happened in our tests, could also be removed

        //instead of clamping just use gradient descent. also works fine and does not require the norm
        Scalarf4 useGD = blend(omega * gradient > Scalarf4(0.0), Scalarf4(1.0), Scalarf4(-1.0));
        omega = Vector3f4::blend(useGD > Scalarf4(0.0), gradient * Scalarf4(-0.125), omega);

        Scalarf4 l_omega2 = omega.lengthSquared();
        const Scalarf4 w = (1.0 - l_omega2) / (1.0 + l_omega2);
        const Vector3f4 vec = omega * (2.0 / (1.0 + l_omega2));
        q = q * Quaternion4f(vec.x(), vec.y(), vec.z(), w);		//no normalization needed because the Cayley map returs a unit quaternion
    }
}

void Physics::solveVolumeConstraints(vector<EigenVector3> &x, const vector<vector<int>> &ind) {

    for (int phase = 0; phase < volume_constraint_phases.size(); phase++)	//forall constraint phases
    {
        for (int constraint = 0; constraint < inv_mass_phases[phase].size(); constraint++)	//forall constraints in this phase
        {
            //move the positions of 4 tetrahedrons to vector registers
            Vector3f4 p[4];

            int c4[4];	//indices of 4 tets
            for (int k = 0; k < 4; k++)
                if (4 * constraint + k < volume_constraint_phases[phase].size())
                    c4[k] = volume_constraint_phases[phase][4 * constraint + k];
                else
                    c4[k] = 0;

            for (int j = 0; j < 4; j++)
            {
                const EigenVector3& p0_0 = x[ind[c4[0]][j]];
                const EigenVector3& p0_1 = x[ind[c4[1]][j]];
                const EigenVector3& p0_2 = x[ind[c4[2]][j]];
                const EigenVector3& p0_3 = x[ind[c4[3]][j]];

                p[j].x() = Scalarf4(p0_0[0], p0_1[0], p0_2[0], p0_3[0]);
                p[j].y() = Scalarf4(p0_0[1], p0_1[1], p0_2[1], p0_3[1]);
                p[j].z() = Scalarf4(p0_0[2], p0_1[2], p0_2[2], p0_3[2]);
            }

            //solve the constraints
            const float eps = 1e-6f;

            //compute the volume using Eq. (14)
            Vector3f4 d1 = p[1] - p[0];
            Vector3f4 d2 = p[2] - p[0];
            Vector3f4 d3 = p[3] - p[0];
            Scalarf4 volume = (d1 % d2) * d3 * (1.0f / 6.0f);

            //compute the gradients (see: supplemental document)
            Vector3f4 grad1 = d2 % d3;
            Vector3f4 grad2 = d3 % d1;
            Vector3f4 grad3 = d1 % d2;
            Vector3f4 grad0 = -grad1 - grad2 - grad3;

            const Scalarf4& restVol = rest_volume_phases[phase][constraint];
            const Scalarf4& alpha = alpha_phases[phase][constraint];
            Scalarf4& kappa = kappa_phases[phase][constraint];

            //compute the Lagrange multiplier update using Eq. (15)
            Scalarf4 delta_kappa =
                    inv_mass_phases[phase][constraint][0] * grad0.lengthSquared() +
                    inv_mass_phases[phase][constraint][1] * grad1.lengthSquared() +
                    inv_mass_phases[phase][constraint][2] * grad2.lengthSquared() +
                    inv_mass_phases[phase][constraint][3] * grad3.lengthSquared() +
                    alpha;

            delta_kappa = (restVol - volume - alpha * kappa) / blend(abs(delta_kappa) < eps, 1.0f, delta_kappa);
            kappa = kappa + delta_kappa;

            //compute the position updates using Eq. (16)
            p[0] = p[0] + grad0 * delta_kappa * inv_mass_phases[phase][constraint][0];
            p[1] = p[1] + grad1 * delta_kappa * inv_mass_phases[phase][constraint][1];
            p[2] = p[2] + grad2 * delta_kappa * inv_mass_phases[phase][constraint][2];
            p[3] = p[3] + grad3 * delta_kappa * inv_mass_phases[phase][constraint][3];

            //write the positions from the vector registers back to the positions array
            for (int j = 0; j < 4; j++)
            {
                float px[4], py[4], pz[4];
                p[j].x().store(px);
                p[j].y().store(py);
                p[j].z().store(pz);

                for (int k = 0; k < 4; k++)
                    if (4 * constraint + k < volume_constraint_phases[phase].size())
                        x[ind[c4[k]][j]] = EigenVector3(px[k], py[k], pz[k]);
            }
        }
    }
}

// getters/setters

void Physics::setGravity(EigenVector3& gravity) {
    this->gravity = gravity;
}

EigenVector3& Physics::getGravity() {
    return this->gravity;
}