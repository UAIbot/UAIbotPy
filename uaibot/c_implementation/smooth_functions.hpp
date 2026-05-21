#pragma once
#include <Eigen/Dense>

#include "declarations.h"

using namespace std;

// ----------------------------------------------------------------------------------------
// Smooth Min / Max functions
// ----------------------------------------------------------------------------------------

float holderMean(float x, float y, float gamma);
Eigen::VectorXf holderMeanGradient(float x, float y, float gamma);
tuple<float, Eigen::VectorXf> holderMeanWithGradient(float x, float y, float gamma);
// Min
float smoothMin2Elements(float x, float y, float gamma);
Eigen::VectorXf smoothMin2ElementsGradient(float x, float y, float gamma);
tuple<float, Eigen::VectorXf> smoothMin2ElementsWithGradient(float x, float y,
                                                             float gamma);
float smoothMinList(const Eigen::VectorXf& values, float gamma);
Eigen::VectorXf smoothMinListGradient(const Eigen::VectorXf& values, float gamma);
tuple<float, Eigen::VectorXf> smoothMinListWithGradient(
    const Eigen::VectorXf& values, float gamma);
// Overloads
float smoothMinList(const std::vector<float>& values, float gamma);
Eigen::VectorXf smoothMinListGradient(const std::vector<float>& values,
                                      float gamma);
tuple<float, Eigen::VectorXf> smoothMinListWithGradient(
    const std::vector<float>& values, float gamma);
// Max
float smoothMax2Elements(float x, float y, float gamma);
Eigen::VectorXf smoothMax2ElementsGradient(float x, float y, float gamma);
tuple<float, Eigen::VectorXf> smoothMax2ElementsWithGradient(float x, float y,
                                                             float gamma);
float smoothMaxList(const Eigen::VectorXf& values, float gamma);
Eigen::VectorXf smoothMaxListGradient(const Eigen::VectorXf& values, float gamma);
tuple<float, Eigen::VectorXf> smoothMaxListWithGradient(
    const Eigen::VectorXf& values, float gamma);
// Overloads
float smoothMaxList(const std::vector<float>& values, float gamma);
Eigen::VectorXf smoothMaxListGradient(const std::vector<float>& values,
                                      float gamma);
tuple<float, Eigen::VectorXf> smoothMaxListWithGradient(
    const std::vector<float>& values, float gamma);

// ----------------------------------------------------------------------------------------
// Auxiliary functions for distance computation
// ----------------------------------------------------------------------------------------

std::vector<Eigen::Vector3f> getBoxVertices(const GeometricPrimitives& box);
std::vector<Eigen::Vector3f> getMinkowskiDifferenceVertices(
    const GeometricPrimitives& box1, const GeometricPrimitives& box2);
std::vector<Eigen::Vector3f> getMinkowskiDifference(
    const std::vector<Eigen::Vector3f>& pointsA,
    const std::vector<Eigen::Vector3f>& pointsB);
std::vector<Eigen::Vector3f> getFaceNormalVectors(
    const GeometricPrimitives& polyhedron);
std::vector<Eigen::Vector3f> getEdgeVectors(
    const GeometricPrimitives& polyhedron);
std::vector<Eigen::Vector3f> getPlatonicSolidEdges(
    const GeometricPrimitives& polyhedron);
std::vector<Eigen::Vector3f> getEdgeNormalVectors(
    const std::vector<Eigen::Vector3f>& edges1,
    const std::vector<Eigen::Vector3f>& edges2);
tuple<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3f>,
      std::vector<Eigen::Vector3f>>
getCandidateNormals(const GeometricPrimitives& polyhedron1,
                    const GeometricPrimitives& polyhedron2,
                    bool isConservative);
tuple<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3f>,
      std::vector<Eigen::Vector3f>>
getCandidateNormals(std::vector<Eigen::Vector3f> faceNormals1,
                    std::vector<Eigen::Vector3f> faceNormals2,
                    std::vector<Eigen::Vector3f> edges1,
                    std::vector<Eigen::Vector3f> edges2, bool isConservative);

std::vector<Eigen::Vector3f> getNormalsVectors(const GeometricPrimitives& box);
float shapingFunction(float u, float k, float epsilon);
std::tuple<float, float> shapingFunctionWithGradient(float u, float k,
                                                     float epsilon);

// ----------------------------------------------------------------------------------------
// Distance functions
// ----------------------------------------------------------------------------------------
tuple<float, Eigen::VectorXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
distBox2Box(const GeometricPrimitives& polyhedron1,
            const GeometricPrimitives& polyhedron2, float gamma,
            bool isConservative = true, bool skipGradient = false, float epsilon = 1e-6f);
tuple<float, Eigen::VectorXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
distSet2Set(const GeometricPrimitives& polyhedron1,
            const GeometricPrimitives& polyhedron2, float gamma,
            bool isConservative = true, bool skipGradient = false, float epsilon = 1e-6f);
tuple<float, Eigen::VectorXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
distSet2Set(std::vector<Eigen::Vector3f> verticesA,
            std::vector<Eigen::Vector3f> verticesB,
            std::vector<Eigen::Vector3f> normalsA,
            std::vector<Eigen::Vector3f> normalsB,
            std::vector<Eigen::Vector3f> edgeNormals, float gamma,
            bool skipGradient = false, float epsilon = 1e-6f);
