#pragma once
#include "declarations.h"
#include <Eigen/Dense>

using namespace std;

// ----------------------------------------------------------------------------------------
// // Smooth Min / Max functions
// ----------------------------------------------------------------------------------------

float holderMean(float x, float y, float r);
Eigen::VectorXf holderMeanGradient(float x, float y, float r);
tuple<float, Eigen::VectorXf> holderMeanWithGradient(float x, float y, float r);
// Min
float smoothMin2Elements(float x, float y, float r);
Eigen::VectorXf smoothMin2ElementsGradient(float x, float y, float r);
tuple<float, Eigen::VectorXf> smoothMin2ElementsWithGradient(float x, float y,
                                                             float r);
float smoothMinList(const Eigen::VectorXf &values, float r);
Eigen::VectorXf smoothMinListGradient(const Eigen::VectorXf &values, float r);
tuple<float, Eigen::VectorXf>
smoothMinListWithGradient(const Eigen::VectorXf &values, float r);
// Overloads
float smoothMinList(const std::vector<float> &values, float r);
Eigen::VectorXf smoothMinListGradient(const std::vector<float> &values,
                                      float r);
tuple<float, Eigen::VectorXf>
smoothMinListWithGradient(const std::vector<float> &values, float r);
// Max
float smoothMax2Elements(float x, float y, float r);
Eigen::VectorXf smoothMax2ElementsGradient(float x, float y, float r);
tuple<float, Eigen::VectorXf> smoothMax2ElementsWithGradient(float x, float y,
                                                             float r);
float smoothMaxList(const Eigen::VectorXf &values, float r);
Eigen::VectorXf smoothMaxListGradient(const Eigen::VectorXf &values, float r);
tuple<float, Eigen::VectorXf>
smoothMaxListWithGradient(const Eigen::VectorXf &values, float r);
// Overloads
float smoothMaxList(const std::vector<float> &values, float r);
Eigen::VectorXf smoothMaxListGradient(const std::vector<float> &values,
                                      float r);
tuple<float, Eigen::VectorXf>
smoothMaxListWithGradient(const std::vector<float> &values, float r);

// ----------------------------------------------------------------------------------------
// // Distance related smooth functions
// ----------------------------------------------------------------------------------------

std::vector<Eigen::Vector3f> getBoxVertices(const GeometricPrimitives &box);
std::vector<Eigen::Vector3f>
getMinkowskiDifferenceVertices(const GeometricPrimitives &box1,
                               const GeometricPrimitives &box2);
std::vector<Eigen::Vector3f> getNormalsVectors(const GeometricPrimitives &box);
tuple<float, Eigen::VectorXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
distBox2Box(const GeometricPrimitives &box1, const GeometricPrimitives &box2,
            float r);
