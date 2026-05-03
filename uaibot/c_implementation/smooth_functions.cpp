#include "smooth_functions.hpp"

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>

// This is uaibot header file
#include "declarations.h"

using namespace std;

// ----------------------------------------------------------------------------------------
// Smooth Min / Max functions
// ----------------------------------------------------------------------------------------

float holderMean(float x, float y, float r) {
  // Eigen::VectorXf powered = values.array().pow(-1.0f / r);
  // Stabler version:
  // If any value is zero, return 0
  if (x == 0.0f || y == 0.0f) {
    return 0.0f;
  }
  // Compute true minimum and 'normalize' values
  float minValue = std::min(x, y);
  float xNorm = x / minValue;
  float yNorm = y / minValue;
  float sumPowered = pow(xNorm, -1.0f / r) + pow(yNorm, -1.0f / r);
  return minValue * pow(sumPowered, -r);
}

Eigen::VectorXf holderMeanGradient(float x, float y, float r) {
  float eps = 1e-6f;
  Eigen::VectorXf gradient(2);
  float dfdx;
  float dfdy;
  // Define cases x=0 and y>0, x>0 and y=0, x=y, and general case
  if (x == 0.0f && y > 0.0f) {
    dfdx = 1.0f;
    dfdy = 0.0f;
  } else if (x > 0.0f && y == 0.0f) {
    dfdx = 0.0f;
    dfdy = 1.0f;
  } else if (abs(x - y) < eps) {
    dfdx = pow(2.0f, -r - 1.0f);
    dfdy = pow(2.0f, -r - 1.0f);
  } else {
    // General case
    dfdx = pow((1.0f + pow(x / y, 1.0f / r)), -r - 1.0f);
    dfdy = pow((1.0f + pow(y / x, 1.0f / r)), -r - 1.0f);
  }
  gradient << dfdx, dfdy;
  // Check if any value in gradient is NaN
  for (Eigen::Index i = 0; i < gradient.size(); ++i) {
    if (isnan(gradient(i))) {
      std::cout << "gradient: " << gradient.transpose() << std::endl;
      // std::cout << "values: " << values.transpose() << std::endl;
      // std::cout << "raised: " << raised.transpose() << std::endl;
      // std::cout << "sumRaised: " << sumRaised << std::endl;
      // std::cout << "outerDer: " << outerDer << std::endl;
      // std::cout << "innerDer: " << innerDer.transpose() << std::endl;
      throw runtime_error("Gradient contains NaN values");
    }
  }
  return gradient;
}

tuple<float, Eigen::VectorXf> holderMeanWithGradient(float x, float y,
                                                     float r) {
  float mean = holderMean(x, y, r);
  Eigen::VectorXf gradient = holderMeanGradient(x, y, r);
  return make_tuple(mean, gradient);
}

// Min
float smoothMin2Elements(float x, float y, float r) {
  if (x >= 0.0f && y >= 0.0f) {
    return holderMean(x, y, r);
  } else if (x < 0.0f && y < 0.0f) {
    float xbar = -1.0f / x;
    float ybar = -1.0f / y;
    float res = holderMean(xbar, ybar, r);
    return -1.0f / res;
  } else {
    return std::min(x, y);
  }
}

Eigen::VectorXf smoothMin2ElementsGradient(float x, float y, float r) {
  if (x >= 0.0f && y >= 0.0f) {
    return holderMeanGradient(x, y, r);
  } else if (x < 0.0f && y < 0.0f) {
    float xbar = -1.0f / x;
    float ybar = -1.0f / y;
    tuple<float, Eigen::VectorXf> res = holderMeanWithGradient(xbar, ybar, r);
    float value = get<0>(res);
    Eigen::VectorXf grad = get<1>(res);
    Eigen::VectorXf chain(2);
    // Avoid near-zero division by adding small epsilon
    float eps = 1e-6f;
    chain << 1.0f / (x * x + eps), 1.0f / (y * y + eps);
    // Apply chain rule d(-1/f(-1/x, -1/y))/dx = ( -1 / f^2 ) * d(-1/x) * df/df
    grad = (grad / (value * value)).cwiseProduct(chain);
    // Check if any value in grad is NaN
    for (Eigen::Index i = 0; i < grad.size(); ++i) {
      if (isnan(grad(i))) {
        std::cout << "values: " << x << ", " << y << std::endl;
        std::cout << "min: " << value << std::endl;
        std::cout << "holder grad: " << get<1>(res).transpose() << std::endl;
        std::cout << "chain: " << chain.transpose() << std::endl;
        std::cout << "grad: " << grad.transpose() << std::endl;
        throw runtime_error("Gradient contains NaN values at pos: " +
                            to_string(i));
      }
    }
    return grad;
  } else {
    Eigen::VectorXf gradient(2);
    if (x < y) {
      gradient << 1.0, 0.0;
    } else {
      gradient << 0.0, 1.0;
    }
    return gradient;
  }
}

tuple<float, Eigen::VectorXf> smoothMin2ElementsWithGradient(float x, float y,
                                                             float r) {
  float value = smoothMin2Elements(x, y, r);
  Eigen::VectorXf gradient = smoothMin2ElementsGradient(x, y, r);
  return make_tuple(value, gradient);
}

float smoothMinList(const Eigen::VectorXf& values, float r) {
  if (values.size() == 0) {
    throw invalid_argument("List of values cannot be empty");
  }
  if (values.size() == 1) {
    return values[0];
  }
  float minValue = values[0];
  for (Eigen::Index i = 1; i < values.size(); ++i) {
    minValue = smoothMin2Elements(minValue, values[i], r);
  }
  return minValue;
}

float smoothMinList(const std::vector<float>& values, float r) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMinList(eigenV, r);
}

Eigen::VectorXf smoothMinListGradient(const Eigen::VectorXf& values, float r) {
  if (values.size() == 0) {
    throw invalid_argument("List of values cannot be empty");
  }
  if (values.size() == 1) {
    Eigen::VectorXf gradient(1);
    gradient << 1.0f;
    return gradient;
  }

  size_t n = values.size();
  Eigen::VectorXf gradient = Eigen::VectorXf::Ones(n);
  float minValue = values[n - 1];

  for (int i = n - 2; i >= 0; --i) {
    tuple<float, Eigen::VectorXf> res =
        smoothMin2ElementsWithGradient(values[i], minValue, r);
    minValue = get<0>(res);
    Eigen::VectorXf localGrad = get<1>(res);
    float left = localGrad(0);
    float right = localGrad(1);
    gradient.segment(i + 1, n - i - 1) *= right;
    gradient(i) *= left;
  }
  return gradient;
}

Eigen::VectorXf smoothMinListGradient(const std::vector<float>& values,
                                      float r) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMinListGradient(eigenV, r);
}

tuple<float, Eigen::VectorXf> smoothMinListWithGradient(
    const Eigen::VectorXf& values, float r) {
  float value = smoothMinList(values, r);
  Eigen::VectorXf gradient = smoothMinListGradient(values, r);
  return make_tuple(value, gradient);
}

tuple<float, Eigen::VectorXf> smoothMinListWithGradient(
    const std::vector<float>& values, float r) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMinListWithGradient(eigenV, r);
}

// Max
float smoothMax2Elements(float x, float y, float r) {
  return -smoothMin2Elements(-x, -y, r);
}

Eigen::VectorXf smoothMax2ElementsGradient(float x, float y, float r) {
  return smoothMin2ElementsGradient(-x, -y, r);
}

tuple<float, Eigen::VectorXf> smoothMax2ElementsWithGradient(float x, float y,
                                                             float r) {
  float value = smoothMax2Elements(x, y, r);
  Eigen::VectorXf gradient = smoothMax2ElementsGradient(x, y, r);
  return make_tuple(value, gradient);
}

float smoothMaxList(const Eigen::VectorXf& values, float r) {
  if (values.size() == 0) {
    throw invalid_argument("List of values cannot be empty");
  }
  if (values.size() == 1) {
    return values[0];
  }
  float maxValue = -smoothMinList(-values, r);
  return maxValue;
}

float smoothMaxList(const std::vector<float>& values, float r) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMaxList(eigenV, r);
}

Eigen::VectorXf smoothMaxListGradient(const Eigen::VectorXf& values, float r) {
  if (values.size() == 0) {
    throw invalid_argument("List of values cannot be empty");
  }
  if (values.size() == 1) {
    Eigen::VectorXf gradient(1);
    gradient << 1.0f;
    return gradient;
  }
  Eigen::VectorXf gradient = smoothMinListGradient(-values, r);
  return gradient;
}

Eigen::VectorXf smoothMaxListGradient(const std::vector<float>& values,
                                      float r) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMaxListGradient(eigenV, r);
}

tuple<float, Eigen::VectorXf> smoothMaxListWithGradient(
    const Eigen::VectorXf& values, float r) {
  float value = smoothMaxList(values, r);
  Eigen::VectorXf gradient = smoothMaxListGradient(values, r);
  return make_tuple(value, gradient);
}

tuple<float, Eigen::VectorXf> smoothMaxListWithGradient(
    const std::vector<float>& values, float r) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMaxListWithGradient(eigenV, r);
}
// ----------------------------------------------------------------------------------------
std::vector<Eigen::Vector3f> getBoxVertices(const GeometricPrimitives& box) {
  if (box.type != 1) {
    throw std::invalid_argument("Input must be a box primitive");
  }
  std::vector<Eigen::Vector3f> vertices(8);
  float half_lx = box.lx / 2.0f;
  float half_ly = box.ly / 2.0f;
  float half_lz = box.lz / 2.0f;

  // Define the 8 vertices of the box in local coordinates
  std::vector<Eigen::Vector3f> local_vertices = {
      Eigen::Vector3f(-half_lx, -half_ly, -half_lz),  // 111
      Eigen::Vector3f(half_lx, -half_ly, -half_lz),   // 011
      Eigen::Vector3f(half_lx, half_ly, -half_lz),    // 001
      Eigen::Vector3f(-half_lx, half_ly, -half_lz),   // 101
      Eigen::Vector3f(-half_lx, -half_ly, half_lz),   // 110
      Eigen::Vector3f(half_lx, -half_ly, half_lz),    // 010
      Eigen::Vector3f(half_lx, half_ly, half_lz),     // 000
      Eigen::Vector3f(-half_lx, half_ly, half_lz)     // 100
  };

  // Transform the local vertices to world coordinates using the box's HTM
  for (int i = 0; i < 8; ++i) {
    vertices[i] = box.htm.block<3, 3>(0, 0) * local_vertices[i] +
                  box.htm.block<3, 1>(0, 3);
  }

  return vertices;
}

// Compute the vertices of minkowski difference of two boxes
std::vector<Eigen::Vector3f> getMinkowskiDifferenceVertices(
    const GeometricPrimitives& box1, const GeometricPrimitives& box2) {
  // The Minkowski difference of two sets A and B is defined as A - B = {a - b |
  // a in A, b in B}.
  if (box1.type != 1 || box2.type != 1) {
    throw std::invalid_argument("Both inputs must be box primitives");
  }
  std::vector<Eigen::Vector3f> vertices_box1 = getBoxVertices(box1);
  std::vector<Eigen::Vector3f> vertices_box2 = getBoxVertices(box2);

  return getMinkowskiDifference(vertices_box1, vertices_box2);
}

// Compute Minkowski difference from points of two sets (not necessarily boxes)
std::vector<Eigen::Vector3f> getMinkowskiDifference(
    const std::vector<Eigen::Vector3f>& pointsA,
    const std::vector<Eigen::Vector3f>& pointsB) {
  std::vector<Eigen::Vector3f> minkowski;
  for (const auto& pA : pointsA) {
    for (const auto& pB : pointsB) {
      minkowski.push_back(pA - pB);
    }
  }
  return minkowski;
}

std::vector<Eigen::Vector3f> getFaceNormalVectors(
    const GeometricPrimitives& polyhedron) {
  // Returns a set with normals and opposite of normals for each face of the
  // polyhedron.
  if (polyhedron.type == 1) {
    // Box case
    std::vector<Eigen::Vector3f> normals(6);
    // The normals of the faces of a box are aligned with the local axes
    // Normal and opposite for face parallel to x-axis
    normals[0] = polyhedron.htm.block<3, 1>(0, 0);
    normals[1] = -polyhedron.htm.block<3, 1>(0, 0);
    // Normal and opposite for face parallel to y-axis
    normals[2] = polyhedron.htm.block<3, 1>(0, 1);
    normals[3] = -polyhedron.htm.block<3, 1>(0, 1);
    // Normal and opposite for face parallel to z-axis
    normals[4] = polyhedron.htm.block<3, 1>(0, 2);
    normals[5] = -polyhedron.htm.block<3, 1>(0, 2);
    return normals;
  } else if (polyhedron.type == 4) {
    // Polytope case
    int rowsA = polyhedron.A.rows();
    std::vector<Eigen::Vector3f> normals(rowsA);
    // Each normal is given by the rows of A, so we add both the row and -row
    // normalized
    for (int i = 0; i < rowsA; i = i + 2) {
      Eigen::Vector3f normal = polyhedron.A.row(i).normalized();
      normals[i] = normal;
      normals[i + 1] = -normal;
    }
    return normals;
  } else {
    throw std::invalid_argument("Unsupported primitive type for normals");
  }
}

std::vector<Eigen::Vector3f> getEdgeVectors(
    const GeometricPrimitives& polyhedron) {
  // Returns a set with edge vectors for each edge of the polyhedron.
  if (polyhedron.type == 1) {
    // Box case
    // The edges of a box are aligned with the local axes
    // We avoid redundant edges, reducing 12->6 edge vectors
    std::vector<Eigen::Vector3f> edges(6);
    // Edge vector parallel to local X
    edges[0] = polyhedron.htm.block<3, 1>(0, 0);   // +X direction
    edges[1] = -polyhedron.htm.block<3, 1>(0, 0);  // -X direction
    // Edge vector parallel to local Y
    edges[2] = polyhedron.htm.block<3, 1>(0, 1);   // +Y direction
    edges[3] = -polyhedron.htm.block<3, 1>(0, 1);  // -Y direction
    // Edge vector parallel to local Z
    edges[4] = polyhedron.htm.block<3, 1>(0, 2);   // +Z direction
    edges[5] = -polyhedron.htm.block<3, 1>(0, 2);  // -Z direction
    return edges;
  } else if (polyhedron.type == 4) {
    // Polytope case
    throw std::invalid_argument(
        "Edge vectors not implemented for polytopes yet");
  } else {
    throw std::invalid_argument("Unsupported primitive type for edge vectors");
  }
}

std::vector<Eigen::Vector3f> getEdgeNormalVectors(
    const std::vector<Eigen::Vector3f>& edges1,
    const std::vector<Eigen::Vector3f>& edges2) {
  // Returns a set with {v, -v} where v is the cross product of each edge of
  // polyhedron1 with each edge of polyhedron2
  int numEdges1 = edges1.size();
  int numEdges2 = edges2.size();
  std::vector<Eigen::Vector3f> crossEdgeNormals(numEdges1 * numEdges2 * 2);
  int idx = 0;
  for (int i = 0; i < numEdges1; ++i) {
    for (int j = 0; j < numEdges2; ++j) {
      Eigen::Vector3f cross = edges1[i].cross(edges2[j]).normalized();
      crossEdgeNormals[idx] = cross;
      crossEdgeNormals[idx + 1] = -cross;
      idx += 2;
    }
  }
  return crossEdgeNormals;
}

tuple<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3f>,
      std::vector<Eigen::Vector3f>>
getCandidateNormals(const GeometricPrimitives& polyhedron1,
                    const GeometricPrimitives& polyhedron2,
                    bool isConservative) {
  std::vector<Eigen::Vector3f> faceNormals1 = getFaceNormalVectors(polyhedron1);
  std::vector<Eigen::Vector3f> faceNormals2 = getFaceNormalVectors(polyhedron2);
  if (isConservative) {
    return make_tuple(faceNormals1, faceNormals2,
                      std::vector<Eigen::Vector3f>());
  } else {
    std::vector<Eigen::Vector3f> edgeVectors1 = getEdgeVectors(polyhedron1);
    std::vector<Eigen::Vector3f> edgeVectors2 = getEdgeVectors(polyhedron2);
    std::vector<Eigen::Vector3f> edgeNormalVectors =
        getEdgeNormalVectors(edgeVectors1, edgeVectors2);
    return make_tuple(faceNormals1, faceNormals2, edgeNormalVectors);
  }
}

tuple<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3f>,
      std::vector<Eigen::Vector3f>>
getCandidateNormals(std::vector<Eigen::Vector3f> faceNormals1,
                    std::vector<Eigen::Vector3f> faceNormals2,
                    std::vector<Eigen::Vector3f> edges1,
                    std::vector<Eigen::Vector3f> edges2, bool isConservative) {
  if (isConservative) {
    return make_tuple(faceNormals1, faceNormals2,
                      std::vector<Eigen::Vector3f>());
  } else {
    std::vector<Eigen::Vector3f> edgeNormalVectors =
        getEdgeNormalVectors(edges1, edges2);
    return make_tuple(faceNormals1, faceNormals2, edgeNormalVectors);
  }
}

tuple<float, Eigen::VectorXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
distBox2Box(const GeometricPrimitives& polyhedron1,
            const GeometricPrimitives& polyhedron2, float gamma,
            bool isConservative) {
  // Throw error if the inputs are not boxes (not Implemented yet)
  if (polyhedron1.type != 1 || polyhedron2.type != 1) {
    throw std::invalid_argument("Both inputs must be box primitives");
  }
  std::vector<Eigen::Vector3f> A = getBoxVertices(polyhedron1);
  std::vector<Eigen::Vector3f> B = getBoxVertices(polyhedron2);
  tuple<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3f>,
        std::vector<Eigen::Vector3f>>
      normalsTuple =
          getCandidateNormals(polyhedron1, polyhedron2, isConservative);
  std::vector<Eigen::Vector3f> normalsA = get<0>(normalsTuple);
  std::vector<Eigen::Vector3f> normalsB = get<1>(normalsTuple);
  std::vector<Eigen::Vector3f> edgeNormals = get<2>(normalsTuple);
  return distSet2Set(A, B, normalsA, normalsB, edgeNormals, gamma);
}

tuple<float, Eigen::VectorXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
distSet2Set(std::vector<Eigen::Vector3f> verticesA,
            std::vector<Eigen::Vector3f> verticesB,
            std::vector<Eigen::Vector3f> normalsA,
            std::vector<Eigen::Vector3f> normalsB,
            std::vector<Eigen::Vector3f> edgeNormals, float gamma) {
  // Returns a tuple with (distance, gradient w.r.t minkowski vertices, gradient
  // w.r.t A vertices, gradient w.r.t B vertices, gradient w.r.t normals) the
  // normals are ordered as [faceNormalsA, edgeNormals, faceNormalsB]
  int numA = verticesA.size();
  int numB = verticesB.size();

  std::vector<Eigen::Vector3f> minkowskiVertices =
      getMinkowskiDifference(verticesA, verticesB);

  // Create normalsSet by concatenating normals of both polyhedra
  std::vector<Eigen::Vector3f> normalsSet;
  normalsSet.insert(normalsSet.end(), normalsA.begin(), normalsA.end());
  // if conservative case, then edgeNormals is empty, so this will not add
  // anything
  normalsSet.insert(normalsSet.end(), edgeNormals.begin(), edgeNormals.end());
  normalsSet.insert(normalsSet.end(), normalsB.begin(), normalsB.end());

  int numN = normalsSet.size();
  int numV = minkowskiVertices.size();

  std::vector<float> innerMins;
  // Matrix with the gradient of the inner minimum with respect to the
  // vertices of the Minkowski difference In paper this would be d(gn)/d(hnc),
  // where hnc=n^T vc for a fixed n and minkowski vertex vc
  Eigen::MatrixXf dGn_dVc(numN, numV);
  for (size_t i = 0; i < numN; ++i) {
    Eigen::Vector3f d = normalsSet[i].normalized();
    std::vector<float> dotProducts(numV);
    for (size_t j = 0; j < numV; ++j) {
      dotProducts[j] = d.dot(minkowskiVertices[j]);
    }
    tuple<float, Eigen::VectorXf> res =
        smoothMinListWithGradient(dotProducts, gamma);
    float dist = get<0>(res);
    Eigen::VectorXf grad = get<1>(res);
    innerMins.push_back(dist);
    // Store the gradient in the rows of the jacobian
    dGn_dVc.row(i) = grad.transpose();  // Size 1 x numV
    // if (i == 0) {
    //   std::cout << "[DEBUG] jacobian row 0 sum: " << jacobian.row(0).sum()
    //             << std::endl;
    //   std::cout << "[DEBUG] jacobian row 0: " << jacobian.row(0) <<
    //   std::endl;
    // }
  }

  tuple<float, Eigen::VectorXf> finalRes =
      smoothMaxListWithGradient(innerMins, gamma);
  float finalDist = get<0>(finalRes);
  Eigen::VectorXf gradSmax = get<1>(finalRes);
  // std::cout << "[DEBUG] gradSmax sum: " << gradSmax.sum()
  //           << ", size: " << gradSmax.size() << std::endl;
  // std::cout << "[DEBUG] gradSmax: " << gradSmax.transpose() << std::endl;
  // Apply chain rule to get the gradient with respect to the Minkowski
  // vertices
  Eigen::VectorXf gradVertsMinkowski = Eigen::VectorXf::Zero(numV);
  gradVertsMinkowski = gradSmax.transpose() * dGn_dVc;  // Size 1 x numV

  // 4. Assemble spatial gradients for A and B vertices
  Eigen::MatrixXf gradVertsA = Eigen::MatrixXf::Zero(numA, 3);
  Eigen::MatrixXf gradVertsB = Eigen::MatrixXf::Zero(numB, 3);
  // Gradient with respect to normals (face directions of A and B)
  Eigen::MatrixXf gradNormals = Eigen::MatrixXf::Zero(numN, 3);

  for (int i = 0; i < numN; ++i) {
    float w_i = gradSmax(i);
    Eigen::Vector3f d_i = normalsSet[i].normalized();

    // For vertices of A
    for (int aIdx = 0; aIdx < numA; ++aIdx) {
      float sumV = 0.0f;
      int base = aIdx * numB;
      for (int bIdx = 0; bIdx < numB; ++bIdx) {
        sumV += dGn_dVc(i, base + bIdx);
      }
      gradVertsA.row(aIdx) += w_i * sumV * d_i.transpose();
    }
    // Eigen::Vector3f sum_grad_box1 = grad_box1.colwise().sum();
    // std::cout << "[DEBUG] sum of grad_box1 over vertices (should match "
    //              "translation gradient): "
    //           << sum_grad_box1.transpose() << std::endl;

    // For vertices of B
    for (int bIdx = 0; bIdx < numB; ++bIdx) {
      float sumV = 0.0f;
      for (int aIdx = 0; aIdx < numA; ++aIdx) {
        sumV += dGn_dVc(i, aIdx * numB + bIdx);
      }
      gradVertsB.row(bIdx) -= w_i * sumV * d_i.transpose();  // minus sign!
    }
    // For normals (face directions)
    for (int vIdx = 0; vIdx < numV; ++vIdx) {
      // w_i * sum_{p in P}sum_{r in R} dgn/dhnpr * (p - r) contribution
      gradNormals.row(i) +=
          w_i * dGn_dVc(i, vIdx) * minkowskiVertices[vIdx].transpose();
    }
  }
  // TODO: Remove this later
  // DEBUGGING STUFF
  // // Lambda to compute scalar distance given box1 vertices P_mod
  // auto computeDistance =
  //     [&](const std::vector<Eigen::Vector3f>& P_mod) -> float {
  //   std::vector<Eigen::Vector3f> minkowski_mod;
  //   minkowski_mod.reserve(numA * numB);
  //   for (const auto& p : P_mod)
  //     for (const auto& r : R) minkowski_mod.push_back(p - r);
  //
  //   std::vector<float> innerMins_mod;
  //   for (size_t i = 0; i < numN; ++i) {
  //     Eigen::Vector3f d = normalsSet[i].normalized();
  //     std::vector<float> dots(numV);
  //     for (size_t j = 0; j < numV; ++j) dots[j] = d.dot(minkowski_mod[j]);
  //     float dist_mod = std::get<0>(smoothMinListWithGradient(dots, r));
  //     innerMins_mod.push_back(dist_mod);
  //   }
  //   return std::get<0>(smoothMaxListWithGradient(innerMins_mod, r));
  // };
  // // ----- DEBUG: Numerical per-vertex gradient check -----
  // float eps = 1e-4f;
  // Eigen::MatrixXf num_grad_box1(num_P, 3);
  // for (int v = 0; v < numA; ++v) {
  //   for (int axis = 0; axis < 3; ++axis) {
  //     Eigen::Vector3f delta = Eigen::Vector3f::Zero();
  //     delta(axis) = eps;
  //
  //     std::vector<Eigen::Vector3f> P_plus = P;
  //     P_plus[v] += delta;
  //     float D_plus = computeDistance(P_plus);
  //
  //     std::vector<Eigen::Vector3f> P_minus = P;
  //     P_minus[v] -= delta;
  //     float D_minus = computeDistance(P_minus);
  //
  //     num_grad_box1(v, axis) = (D_plus - D_minus) / (2.0f * eps);
  //   }
  // }
  // std::cout << "[DEBUG] Analytical grad_box1:\n" << grad_box1 << std::endl;
  // std::cout << "[DEBUG] Numerical grad_box1:\n" << num_grad_box1 <<
  // std::endl; std::cout << "[DEBUG] Difference:\n"
  //           << (grad_box1 - num_grad_box1) << std::endl;
  // // --------------------------------------------------------
  // Eigen::Vector3f ana_grad_t = grad_box1.colwise().sum();
  // Eigen::Vector3f num_grad_t = num_grad_box1.colwise().sum();
  // std::cout << "[DEBUG] Analytical translation grad: " <<
  // ana_grad_t.transpose()
  //           << std::endl;
  // std::cout << "[DEBUG] Numerical  translation grad: " <<
  // num_grad_t.transpose()
  //           << std::endl;
  //
  // ----- DEBUG: Numerical normal gradient check (rotation-based) -----
  // auto computeDistanceWithNormal = [&](int normal_idx,
  //                                      const Eigen::Vector3f& n_mod) ->
  //                                      float
  //                                      {
  //   // Recompute only the smooth min for the modified normal, then redo
  //   smooth
  //   // max
  //   std::vector<float> innerMins_mod = innerMins;  // copy original inner
  //   mins
  //   // Recompute the entry for the modified normal
  //   Eigen::Vector3f d_mod = n_mod.normalized();
  //   std::vector<float> dots_mod(numV);
  //   for (int j = 0; j < numV; ++j) {
  //     dots_mod[j] = d_mod.dot(minkowskiVertices[j]);
  //   }
  //   innerMins_mod[normal_idx] =
  //       std::get<0>(smoothMinListWithGradient(dots_mod, r));
  //   return std::get<0>(smoothMaxListWithGradient(innerMins_mod, r));
  // };
  // Eigen::MatrixXf num_grad_normals(num_N, 3);
  // float rot_eps = 1e-4f;  // small rotation angle in radians
  //
  // for (int i = 0; i < numN; ++i) {
  //   Eigen::Vector3f n_orig = normalsSet[i].normalized();
  //
  //   // Build two orthonormal tangent vectors
  //   Eigen::Vector3f t1, t2;
  //   if (std::abs(n_orig.x()) > std::abs(n_orig.y()))
  //     t1 = Eigen::Vector3f(-n_orig.z(), 0.0f, n_orig.x()).normalized();
  //   else
  //     t1 = Eigen::Vector3f(0.0f, n_orig.z(), -n_orig.y()).normalized();
  //   t2 = n_orig.cross(t1).normalized();
  //
  //   // Lambda to rotate n_orig around axis 'u' by angle 'theta'
  //   auto rotate = [&](const Eigen::Vector3f& u,
  //                     float theta) -> Eigen::Vector3f {
  //     Eigen::AngleAxisf rot(theta, u);
  //     return rot * n_orig;
  //   };
  //
  //   // Numerical derivatives w.r.t. rotation around t1 and t2
  //   float D_plus_t1 = computeDistanceWithNormal(i, rotate(t1, rot_eps));
  //   float D_minus_t1 = computeDistanceWithNormal(i, rotate(t1, -rot_eps));
  //   float dD_dtheta1 = (D_plus_t1 - D_minus_t1) / (2.0f * rot_eps);
  //
  //   float D_plus_t2 = computeDistanceWithNormal(i, rotate(t2, rot_eps));
  //   float D_minus_t2 = computeDistanceWithNormal(i, rotate(t2, -rot_eps));
  //   float dD_dtheta2 = (D_plus_t2 - D_minus_t2) / (2.0f * rot_eps);
  //
  //   // The relation: dD = (∂D/∂n)·dn, with dn = (u × n) dθ.
  //   // So dD/dθ = (∂D/∂n)·(u × n).
  //   // For u = t1, u × n = t2 (since t1,t2,n form right-handed orthonormal)
  //   // For u = t2, u × n = -t1.
  //   // Therefore:
  //   // dD_dtheta1 = grad_n · t2
  //   // dD_dtheta2 = grad_n · (-t1)
  //   // Solve for grad_n (projected onto tangent plane):
  //   Eigen::Vector3f num_grad_tangent = dD_dtheta1 * t2 - dD_dtheta2 * t1;
  //
  //   // The full analytical gradient (may have a component along n, but that
  //   is
  //   // zero for unit vector constraint) We only compare the tangential
  //   part. num_grad_normals.row(i) = num_grad_tangent.transpose();
  // }
  //
  // // For comparison, project analytical gradient onto tangent plane
  // Eigen::MatrixXf ana_grad_tangent(num_N, 3);
  // for (int i = 0; i < numN; ++i) {
  //   Eigen::Vector3f n = normalsSet[i].normalized();
  //   Eigen::Vector3f ana = grad_normals.row(i).transpose();
  //   // Remove component along n (should be zero anyway, but for safety)
  //   Eigen::Vector3f ana_tan = ana - (ana.dot(n)) * n;
  //   ana_grad_tangent.row(i) = ana_tan.transpose();
  // }
  //
  // std::cout << "[DEBUG] Analytical grad_normals (tangential part):\n"
  //           << ana_grad_tangent << std::endl;
  // std::cout << "[DEBUG] Numerical grad_normals (tangential):\n"
  //           << num_grad_normals << std::endl;
  // std::cout << "[DEBUG] Difference (tangential):\n"
  //           << (ana_grad_tangent - num_grad_normals) << std::endl;
  // -------------------------------------------------------------
  return make_tuple(finalDist, gradVertsMinkowski, gradVertsA, gradVertsB, gradNormals);
}
