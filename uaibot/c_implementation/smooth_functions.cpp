#include "smooth_functions.hpp"

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <ostream>

// This is uaibot header file
#include "declarations.h"

using namespace std;

// ----------------------------------------------------------------------------------------
// Smooth Min / Max functions
// ----------------------------------------------------------------------------------------

float holderMean(float x, float y, float gamma) {
  float r = 1 / (gamma +
                 1);  // Old variable was 0 < r < 1, but integer gamma is easier
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

Eigen::VectorXf holderMeanGradient(float x, float y, float gamma) {
  float r = 1 / (gamma +
                 1);  // Old variable was 0 < r < 1, but integer gamma is easier
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
                                                     float gamma) {
  float mean = holderMean(x, y, gamma);
  Eigen::VectorXf gradient = holderMeanGradient(x, y, gamma);
  return make_tuple(mean, gradient);
}

// Min
float smoothMin2Elements(float x, float y, float gamma) {
  if (x >= 0.0f && y >= 0.0f) {
    return holderMean(x, y, gamma);
  } else if (x < 0.0f && y < 0.0f) {
    float xbar = -1.0f / x;
    float ybar = -1.0f / y;
    float res = holderMean(xbar, ybar, gamma);
    return -1.0f / res;
  } else {
    return std::min(x, y);
  }
}

Eigen::VectorXf smoothMin2ElementsGradient(float x, float y, float gamma) {
  if (x >= 0.0f && y >= 0.0f) {
    return holderMeanGradient(x, y, gamma);
  } else if (x < 0.0f && y < 0.0f) {
    float xbar = -1.0f / x;
    float ybar = -1.0f / y;
    tuple<float, Eigen::VectorXf> res =
        holderMeanWithGradient(xbar, ybar, gamma);
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
                                                             float gamma) {
  float value = smoothMin2Elements(x, y, gamma);
  Eigen::VectorXf gradient = smoothMin2ElementsGradient(x, y, gamma);
  return make_tuple(value, gradient);
}

float smoothMinList(const Eigen::VectorXf& values, float gamma) {
  if (values.size() == 0) {
    throw invalid_argument("List of values cannot be empty");
  }
  if (values.size() == 1) {
    return values[0];
  }
  float minValue = values[0];
  for (Eigen::Index i = 1; i < values.size(); ++i) {
    minValue = smoothMin2Elements(minValue, values[i], gamma);
  }
  return minValue;
}

float smoothMinList(const std::vector<float>& values, float gamma) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMinList(eigenV, gamma);
}

Eigen::VectorXf smoothMinListGradient(const Eigen::VectorXf& values,
                                      float gamma) {
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
        smoothMin2ElementsWithGradient(values[i], minValue, gamma);
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
                                      float gamma) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMinListGradient(eigenV, gamma);
}

tuple<float, Eigen::VectorXf> smoothMinListWithGradient(
    const Eigen::VectorXf& values, float gamma) {
  float value = smoothMinList(values, gamma);
  Eigen::VectorXf gradient = smoothMinListGradient(values, gamma);
  return make_tuple(value, gradient);
}

tuple<float, Eigen::VectorXf> smoothMinListWithGradient(
    const std::vector<float>& values, float gamma) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMinListWithGradient(eigenV, gamma);
}

// Max
float smoothMax2Elements(float x, float y, float gamma) {
  return -smoothMin2Elements(-x, -y, gamma);
}

Eigen::VectorXf smoothMax2ElementsGradient(float x, float y, float gamma) {
  return smoothMin2ElementsGradient(-x, -y, gamma);
}

tuple<float, Eigen::VectorXf> smoothMax2ElementsWithGradient(float x, float y,
                                                             float gamma) {
  float value = smoothMax2Elements(x, y, gamma);
  Eigen::VectorXf gradient = smoothMax2ElementsGradient(x, y, gamma);
  return make_tuple(value, gradient);
}

float smoothMaxList(const Eigen::VectorXf& values, float gamma) {
  if (values.size() == 0) {
    throw invalid_argument("List of values cannot be empty");
  }
  if (values.size() == 1) {
    return values[0];
  }
  float maxValue = -smoothMinList(-values, gamma);
  return maxValue;
}

float smoothMaxList(const std::vector<float>& values, float gamma) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMaxList(eigenV, gamma);
}

Eigen::VectorXf smoothMaxListGradient(const Eigen::VectorXf& values,
                                      float gamma) {
  if (values.size() == 0) {
    throw invalid_argument("List of values cannot be empty");
  }
  if (values.size() == 1) {
    Eigen::VectorXf gradient(1);
    gradient << 1.0f;
    return gradient;
  }
  Eigen::VectorXf gradient = smoothMinListGradient(-values, gamma);
  return gradient;
}

Eigen::VectorXf smoothMaxListGradient(const std::vector<float>& values,
                                      float gamma) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMaxListGradient(eigenV, gamma);
}

tuple<float, Eigen::VectorXf> smoothMaxListWithGradient(
    const Eigen::VectorXf& values, float gamma) {
  float value = smoothMaxList(values, gamma);
  Eigen::VectorXf gradient = smoothMaxListGradient(values, gamma);
  return make_tuple(value, gradient);
}

tuple<float, Eigen::VectorXf> smoothMaxListWithGradient(
    const std::vector<float>& values, float gamma) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMaxListWithGradient(eigenV, gamma);
}
// ----------------------------------------------------------------------------------------
// Auxiliary functions for distance computation
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

std::vector<Eigen::Vector3f> getPlatonicSolidEdges(
    const GeometricPrimitives& polyhedron) {
  // This function should return the edge vectors for platonic solids (type 4)
  // For simplicity, we can hardcode the edge vectors for the 5 platonic solids
  // based on their vertices and faces.
  if (polyhedron.type != 4) {
    throw std::invalid_argument("Input must be a polytope primitive");
  }
  // Placeholder: return an empty vector for now
  int numFaces = polyhedron.A.rows();
  Eigen::Matrix4f htm = polyhedron.htm;
  Eigen::Matrix3f R = htm.block<3, 3>(0, 0);
  // Helper: given a set of vertices (in local frame), returns a unit vector
  // for each edge (the directed vector from one vertex to its neighbour).
  auto edges_from_vertices = [](const std::vector<Eigen::Vector3f>& vertices)
      -> std::vector<Eigen::Vector3f> {
    std::vector<Eigen::Vector3f> edges;
    const int n = static_cast<int>(vertices.size());
    if (n < 2) return edges;

    // Find edge length as the smallest positive squared distance
    float edge_len_sq = std::numeric_limits<float>::max();
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) {
        float d2 = (vertices[j] - vertices[i]).squaredNorm();
        if (d2 > 1e-6f && d2 < edge_len_sq) edge_len_sq = d2;
      }

    float tol =
        edge_len_sq * 1e-4f;  // relative tolerance for distance equality
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) {
        float d2 = (vertices[j] - vertices[i]).squaredNorm();
        if (std::abs(d2 - edge_len_sq) < tol)
          edges.push_back((vertices[j] - vertices[i]).normalized());
      }
    return edges;
  };

  std::vector<Eigen::Vector3f> canonical_edges;

  if (numFaces == 4) {
    // Tetrahedron (vertices: (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1))
    canonical_edges = edges_from_vertices({{1.0f, 1.0f, 1.0f},
                                           {1.0f, -1.0f, -1.0f},
                                           {-1.0f, 1.0f, -1.0f},
                                           {-1.0f, -1.0f, 1.0f}});
  } else if (numFaces == 6) {
    // Cube (vertices: all ±1)
    canonical_edges = edges_from_vertices({{-1.0f, -1.0f, -1.0f},
                                           {-1.0f, -1.0f, 1.0f},
                                           {-1.0f, 1.0f, -1.0f},
                                           {-1.0f, 1.0f, 1.0f},
                                           {1.0f, -1.0f, -1.0f},
                                           {1.0f, -1.0f, 1.0f},
                                           {1.0f, 1.0f, -1.0f},
                                           {1.0f, 1.0f, 1.0f}});
  } else if (numFaces == 8) {
    // Octahedron (vertices on axes)
    canonical_edges = edges_from_vertices({{1.0f, 0.0f, 0.0f},
                                           {-1.0f, 0.0f, 0.0f},
                                           {0.0f, 1.0f, 0.0f},
                                           {0.0f, -1.0f, 0.0f},
                                           {0.0f, 0.0f, 1.0f},
                                           {0.0f, 0.0f, -1.0f}});
  } else if (numFaces == 12) {
    // Dodecahedron (20 vertices)
    const float phi = (1.0f + std::sqrt(5.0f)) / 2.0f;
    const float inv_phi = 1.0f / phi;
    std::vector<Eigen::Vector3f> verts;

    // 8 vertices: (±1, ±1, ±1)
    for (float x : {-1.0f, 1.0f})
      for (float y : {-1.0f, 1.0f})
        for (float z : {-1.0f, 1.0f}) verts.push_back({x, y, z});

    // 12 vertices: cyclic permutations of (0, ±φ, ±1/φ)
    for (float a : {-phi, phi})
      for (float b : {-inv_phi, inv_phi}) {
        verts.push_back({0.0f, a, b});
        verts.push_back({b, 0.0f, a});
        verts.push_back({a, b, 0.0f});
      }

    canonical_edges = edges_from_vertices(verts);
  } else if (numFaces == 20) {
    // Icosahedron (12 vertices)
    const float phi = (1.0f + std::sqrt(5.0f)) / 2.0f;
    std::vector<Eigen::Vector3f> verts;

    // Cyclic permutations of (0, ±1, ±φ)
    for (float b : {-1.0f, 1.0f})
      for (float c : {-phi, phi}) {
        verts.push_back({0.0f, b, c});
        verts.push_back({c, 0.0f, b});
        verts.push_back({b, c, 0.0f});
      }

    canonical_edges = edges_from_vertices(verts);
  } else {
    throw std::invalid_argument(
        "Unsupported number of faces for platonic solid: " +
        std::to_string(numFaces));
  }

  // Rotate all edge directions into world frame
  std::vector<Eigen::Vector3f> world_edges;
  world_edges.reserve(canonical_edges.size());
  for (const auto& v : canonical_edges) {
    world_edges.push_back((R * v).normalized());
  }
  return world_edges;
}

std::vector<Eigen::Vector3f> getEdgeVectors(
    const GeometricPrimitives& polyhedron) {
  // Returns a set with edge vectors for each edge of the polyhedron.
  if (polyhedron.type == 1) {
    // Box case
    // The edges of a box are aligned with the local axes
    std::vector<Eigen::Vector3f> edges(12);
    // Edge vector parallel to local X
    edges[0] = polyhedron.htm.block<3, 1>(0, 0);   // +X direction
    edges[1] = polyhedron.htm.block<3, 1>(0, 0);
    edges[2] = polyhedron.htm.block<3, 1>(0, 0);
    edges[3] = polyhedron.htm.block<3, 1>(0, 0);
    // Edge vector parallel to local Y
    edges[4] = polyhedron.htm.block<3, 1>(0, 1);   // +Y direction
    edges[5] = polyhedron.htm.block<3, 1>(0, 1);
    edges[6] = polyhedron.htm.block<3, 1>(0, 1);
    edges[7] = polyhedron.htm.block<3, 1>(0, 1);
    // Edge vector parallel to local Z
    edges[8] = polyhedron.htm.block<3, 1>(0, 2);   // +Z direction
    edges[9] = polyhedron.htm.block<3, 1>(0, 2);
    edges[10] = polyhedron.htm.block<3, 1>(0, 2);
    edges[11] = polyhedron.htm.block<3, 1>(0, 2);
    return edges;
  } else if (polyhedron.type == 4) {
    // This currently only supports platonic solids
    return getPlatonicSolidEdges(polyhedron);

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
  // TODO: testing Edges only scenario
  std::vector<Eigen::Vector3f> crossEdgeNormals;
  crossEdgeNormals.reserve(numEdges1 + numEdges2);  // optional, avoids reallocation
  crossEdgeNormals.insert(crossEdgeNormals.end(), edges1.begin(), edges1.end());
  crossEdgeNormals.insert(crossEdgeNormals.end(), edges2.begin(), edges2.end());

  // Uncomment to revert
  // std::vector<Eigen::Vector3f> crossEdgeNormals(numEdges1 * numEdges2 * 2);
  // int idx = 0;
  // for (int i = 0; i < numEdges1; ++i) {
  //   for (int j = 0; j < numEdges2; ++j) {
  //     Eigen::Vector3f cross = edges1[i].cross(edges2[j]).normalized();
  //     crossEdgeNormals[idx] = cross;
  //     crossEdgeNormals[idx + 1] = -cross;
  //     idx += 2;
  //   }
  // }
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

float shapingFunction(float u, float k, float epsilon) {
  /* A k-th order shaping function that removes non-differentiability at 0,
   * defined as: phi(u) = u * (|u|^k) / (|u|^k + epsilon)
   */
  float abs_u_k = pow(abs(u), k);
  float phi = u * (abs_u_k / (abs_u_k + epsilon));
  phi = std::isnan(phi) ? 0.0f : phi;
  return phi;
}
std::tuple<float, float> shapingFunctionWithGradient(float u, float k,
                                                     float epsilon) {
  /* Returns the k-th order shaping function and its gradient w.r.t u, defined as:
   *  phi(u) = u * (|u|^k) / (|u|^k + epsilon)
   */
  float abs_u = abs(u);
  float abs_u_k = pow(abs_u, k);
  float u_squared = u * u;
  float abs_u_k_minus_2 = pow(abs_u, k - 2);
  float abs_u_k_plus_2 = pow(abs_u, k + 2);
  float eps_pow = pow(abs_u_k + epsilon, 2);
  float dphi_du = 0.0;
  if (eps_pow > 1e-6) {
    float dphi_du =
        (abs_u_k_minus_2 * (abs_u_k_plus_2 + (k + 1) * epsilon * u_squared)) /
        eps_pow;
  }
  float phi = u * (abs_u_k / (abs_u_k + epsilon));
  phi = std::isnan(phi) ? 0.0f : phi;
  return make_tuple(phi, dphi_du);
}

// ----------------------------------------------------------------------------------------
// Distance functions
// ----------------------------------------------------------------------------------------
tuple<float, Eigen::VectorXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
distBox2Box(const GeometricPrimitives& polyhedron1,
            const GeometricPrimitives& polyhedron2, float gamma,
            bool isConservative, bool skipGradient, float epsilon) {
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
  return distSet2Set(A, B, normalsA, normalsB, edgeNormals, gamma, skipGradient,
                     epsilon);
}

tuple<float, Eigen::VectorXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
distSet2Set(const GeometricPrimitives& polyhedron1,
            const GeometricPrimitives& polyhedron2, float gamma,
            bool isConservative, bool skipGradient, float epsilon) {
  // Throw errors if type is different than 1 or 4 (not Implemented yet for
  // non-polyhedra)
  if ((polyhedron1.type != 1 && polyhedron1.type != 4) ||
      (polyhedron2.type != 1 && polyhedron2.type != 4)) {
    throw std::invalid_argument(
        "Both inputs must be either box primitives or polytope primitives");
  }
  std::vector<Eigen::Vector3f> vertices1_local = polyhedron1.vertices_local;
  std::vector<Eigen::Vector3f> vertices2_local = polyhedron2.vertices_local;
  // Get A, B as the world vertices of the two polyhedra
  Eigen::Matrix4f htm1 = polyhedron1.htm;
  Eigen::Matrix4f htm2 = polyhedron2.htm;
  std::vector<Eigen::Vector3f> A, B;
  A.reserve(vertices1_local.size());
  B.reserve(vertices2_local.size());

  for (const auto& v : vertices1_local) {
    Eigen::Vector4f vh(v.x(), v.y(), v.z(), 1.0f);
    Eigen::Vector4f wh = htm1 * vh;
    A.push_back(wh.head<3>());
  }
  for (const auto& v : vertices2_local) {
    Eigen::Vector4f vh(v.x(), v.y(), v.z(), 1.0f);
    Eigen::Vector4f wh = htm2 * vh;
    B.push_back(wh.head<3>());
  }
  tuple<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3f>,
        std::vector<Eigen::Vector3f>>
      normalsTuple =
          getCandidateNormals(polyhedron1, polyhedron2, isConservative);
  std::vector<Eigen::Vector3f> normalsA = get<0>(normalsTuple);
  std::vector<Eigen::Vector3f> normalsB = get<1>(normalsTuple);
  std::vector<Eigen::Vector3f> edgeNormals = get<2>(normalsTuple);
  return distSet2Set(A, B, normalsA, normalsB, edgeNormals, gamma, skipGradient,
                     epsilon);
}

tuple<float, Eigen::VectorXf, Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf>
distSet2Set(std::vector<Eigen::Vector3f> verticesA,
            std::vector<Eigen::Vector3f> verticesB,
            std::vector<Eigen::Vector3f> normalsA,
            std::vector<Eigen::Vector3f> normalsB,
            std::vector<Eigen::Vector3f> edgeNormals, float gamma,
            bool skipGradient, float epsilon) {
  // Returns a tuple with (distance, gradient w.r.t minkowski vertices, gradient
  // w.r.t A vertices, gradient w.r.t B vertices, gradient w.r.t normals) the
  // normals are ordered as [faceNormalsA, edgeNormals, faceNormalsB]
  int numA = verticesA.size();
  int numB = verticesB.size();
  // TODO: remove this when gamma is updated
  // previously 0 < gamma < 1, now we set gamma = 1/(gamma'+1) and use gamma'
  // shaping function uses gamma'
  // float gamma_mod = (1 - gamma) / gamma;

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
    // std::cout << "[DEBUG] inner min at N=" << i << std::endl;
    if (skipGradient) {
      float dist = smoothMinList(dotProducts, gamma);
      float dist_mod = shapingFunction(dist, gamma, epsilon);
      // std::cout << "min= " << dist << " phi(dist())= " << dist_mod << std::endl;
      innerMins.push_back(dist_mod);
    } else {
      tuple<float, Eigen::VectorXf> res =
          smoothMinListWithGradient(dotProducts, gamma);
      float dist = get<0>(res);
      Eigen::VectorXf grad = get<1>(res);
      tuple<float, float> res_mod =
          shapingFunctionWithGradient(dist, gamma, epsilon);
      float dist_mod = get<0>(res_mod);
      float dphi_du = get<1>(res_mod);
      innerMins.push_back(dist_mod);
      // Store the gradient in the rows of the jacobian
      dGn_dVc.row(i) = grad.transpose() * dphi_du;  // Size 1 x numV
      // if (i == 0) {
      //   std::cout << "[DEBUG] jacobian row 0 sum: " << jacobian.row(0).sum()
      //             << std::endl;
      //   std::cout << "[DEBUG] jacobian row 0: " << jacobian.row(0) <<
      //   std::endl;
      // }
    }
  }

  if (skipGradient) {
    float finalDist = smoothMaxList(innerMins, gamma);
    float finalDist_mod = shapingFunction(finalDist, gamma, epsilon);
    return make_tuple(finalDist_mod, Eigen::VectorXf(), Eigen::MatrixXf(),
                      Eigen::MatrixXf(), Eigen::MatrixXf());
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
  tuple<float, float> res_mod =
      shapingFunctionWithGradient(finalDist, gamma, epsilon);
  // For simplicity, we change the variables themselves to avoid changing the
  // other loops
  float Deletethisvariable = finalDist;
  finalDist = get<0>(res_mod);
  float dphi_du = get<1>(res_mod);
  gradSmax *= dphi_du;
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
  return make_tuple(finalDist, gradVertsMinkowski, gradVertsA, gradVertsB,
                    gradNormals);
}
