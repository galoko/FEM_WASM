#pragma once

#include <Eigen/Dense>
#include <cfloat>
#define _USE_MATH_DEFINES
#include <cmath> 

using EigenVector3 = Eigen::Matrix<float, 3, 1>;
using EigenMatrix3 = Eigen::Matrix<float, 3, 3>;
using EigenMatrix4 = Eigen::Matrix<float, 4, 4>;
using EigenAngleAxis = Eigen::AngleAxis<float>;
using EigenQuaternion = Eigen::Quaternion<float>;