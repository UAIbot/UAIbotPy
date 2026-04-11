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

float smoothMinList(const Eigen::VectorXf &values, float r) {
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

float smoothMinList(const std::vector<float> &values, float r) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMinList(eigenV, r);
}

Eigen::VectorXf smoothMinListGradient(const Eigen::VectorXf &values, float r) {
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

Eigen::VectorXf smoothMinListGradient(const std::vector<float> &values,
                                      float r) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMinListGradient(eigenV, r);
}

tuple<float, Eigen::VectorXf>
smoothMinListWithGradient(const Eigen::VectorXf &values, float r) {
  float value = smoothMinList(values, r);
  Eigen::VectorXf gradient = smoothMinListGradient(values, r);
  return make_tuple(value, gradient);
}

tuple<float, Eigen::VectorXf>
smoothMinListWithGradient(const std::vector<float> &values, float r) {
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

float smoothMaxList(const Eigen::VectorXf &values, float r) {
  if (values.size() == 0) {
    throw invalid_argument("List of values cannot be empty");
  }
  if (values.size() == 1) {
    return values[0];
  }
  float maxValue = -smoothMinList(-values, r);
  return maxValue;
}

float smoothMaxList(const std::vector<float> &values, float r) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMaxList(eigenV, r);
}

Eigen::VectorXf smoothMaxListGradient(const Eigen::VectorXf &values, float r) {
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

Eigen::VectorXf smoothMaxListGradient(const std::vector<float> &values,
                                      float r) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMaxListGradient(eigenV, r);
}

tuple<float, Eigen::VectorXf>
smoothMaxListWithGradient(const Eigen::VectorXf &values, float r) {
  float value = smoothMaxList(values, r);
  Eigen::VectorXf gradient = smoothMaxListGradient(values, r);
  return make_tuple(value, gradient);
}

tuple<float, Eigen::VectorXf>
smoothMaxListWithGradient(const std::vector<float> &values, float r) {
  // Convert std::vector<float> to Eigen::VectorXf and call the other function
  Eigen::Map<const Eigen::VectorXf> eigenV(values.data(), values.size());
  return smoothMaxListWithGradient(eigenV, r);
}
// ----------------------------------------------------------------------------------------
std::vector<Eigen::Vector3f> getBoxVertices(const GeometricPrimitives &box) {
  if (box.type != 1) {
    throw std::invalid_argument("Input must be a box primitive");
  }
  std::vector<Eigen::Vector3f> vertices(8);
  float half_lx = box.lx / 2.0f;
  float half_ly = box.ly / 2.0f;
  float half_lz = box.lz / 2.0f;

  // Define the 8 vertices of the box in local coordinates
  std::vector<Eigen::Vector3f> local_vertices = {
      Eigen::Vector3f(-half_lx, -half_ly, -half_lz), // 111
      Eigen::Vector3f(half_lx, -half_ly, -half_lz),  // 011
      Eigen::Vector3f(half_lx, half_ly, -half_lz),   // 001
      Eigen::Vector3f(-half_lx, half_ly, -half_lz),  // 101
      Eigen::Vector3f(-half_lx, -half_ly, half_lz),  // 110
      Eigen::Vector3f(half_lx, -half_ly, half_lz),   // 010
      Eigen::Vector3f(half_lx, half_ly, half_lz),    // 000
      Eigen::Vector3f(-half_lx, half_ly, half_lz)    // 100
  };

  // Transform the local vertices to world coordinates using the box's HTM
  for (int i = 0; i < 8; ++i) {
    vertices[i] = box.htm.block<3, 3>(0, 0) * local_vertices[i] +
                  box.htm.block<3, 1>(0, 3);
  }

  return vertices;
}

// Compute the vertices of minkowski difference of two boxes
std::vector<Eigen::Vector3f>
getMinkowskiDifferenceVertices(const GeometricPrimitives &box1,
                               const GeometricPrimitives &box2) {
  // The Minkowski difference of two sets A and B is defined as A - B = {a - b |
  // a in A, b in B}.
  if (box1.type != 1 || box2.type != 1) {
    throw std::invalid_argument("Both inputs must be box primitives");
  }
  std::vector<Eigen::Vector3f> vertices_box1 = getBoxVertices(box1);
  std::vector<Eigen::Vector3f> vertices_box2 = getBoxVertices(box2);
  std::vector<Eigen::Vector3f> minkowski_vertices;

  for (const auto &v1 : vertices_box1) {
    for (const auto &v2 : vertices_box2) {
      minkowski_vertices.push_back(v1 - v2);
    }
  }
  return minkowski_vertices;
}

// Compute the set of normal vectors (currently not named)
std::vector<Eigen::Vector3f> getNormalsVectors(const GeometricPrimitives &box) {
  // This is a simplified version that assumes that only the face vectors are
  // necessary. It is more conservative than using the complete set
  std::vector<Eigen::Vector3f> normals(6);
  // The normals of the faces of a box are aligned with the local axes
  normals[0] = box.htm.block<3, 1>(0, 0); // Normal for face parallel to x-axis
  normals[1] =
      -box.htm.block<3, 1>(0, 0); // Opposite normal for face parallel to x-axis
  normals[2] = box.htm.block<3, 1>(0, 1); // Normal for face parallel to y-axis
  normals[3] =
      -box.htm.block<3, 1>(0, 1); // Opposite normal for face parallel to y-axis
  normals[4] = box.htm.block<3, 1>(0, 2); // Normal for face parallel to z-axis
  normals[5] =
      -box.htm.block<3, 1>(0, 2); // Opposite normal for face parallel to z-axis
  return normals;
}

tuple<float, Eigen::VectorXf, Eigen::MatrixXf, Eigen::MatrixXf>
distBox2Box(const GeometricPrimitives &box1, const GeometricPrimitives &box2,
            float r) {
  // Throw error if the inputs are not boxes (not Implemented yet)
  if (box1.type != 1 || box2.type != 1) {
    throw std::invalid_argument("Both inputs must be box primitives");
  }
  std::vector<Eigen::Vector3f> P =
      getBoxVertices(box1); // box1 vertices (size num_P)
  std::vector<Eigen::Vector3f> R =
      getBoxVertices(box2); // box2 vertices (size num_R)
  int num_P = P.size();
  int num_R = R.size();
  std::vector<Eigen::Vector3f> minkowskiVertices =
      getMinkowskiDifferenceVertices(box1, box2);
  std::vector<Eigen::Vector3f> normalsBox1 = getNormalsVectors(box1);
  std::vector<Eigen::Vector3f> normalsBox2 = getNormalsVectors(box2);
  // Create normalsSet by concatenating normals of both boxes
  std::vector<Eigen::Vector3f> normalsSet;
  normalsSet.insert(normalsSet.end(), normalsBox1.begin(), normalsBox1.end());
  normalsSet.insert(normalsSet.end(), normalsBox2.begin(), normalsBox2.end());
  int num_N = normalsSet.size();
  int num_V = minkowskiVertices.size();
  std::vector<float> innerMins;
  Eigen::MatrixXf jacobian(num_N, num_V);
  for (size_t i = 0; i < num_N; ++i) {
    Eigen::Vector3f d = normalsSet[i].normalized();
    std::vector<float> dotProducts(num_V);
    for (size_t j = 0; j < num_V; ++j) {
      dotProducts[j] = d.dot(minkowskiVertices[j]);
    }
    tuple<float, Eigen::VectorXf> res =
        smoothMinListWithGradient(dotProducts, r);
    float dist = get<0>(res);
    Eigen::VectorXf grad = get<1>(res);
    innerMins.push_back(dist);
    // Store the gradient in the rows of the jacobian
    jacobian.row(i) = grad.transpose(); // Size 1 x num_V
    // if (i == 0) {
    //   std::cout << "[DEBUG] jacobian row 0 sum: " << jacobian.row(0).sum()
    //             << std::endl;
    //   std::cout << "[DEBUG] jacobian row 0: " << jacobian.row(0) << std::endl;
    // }
  }

  tuple<float, Eigen::VectorXf> finalRes =
      smoothMaxListWithGradient(innerMins, r);
  float finalDist = get<0>(finalRes);
  Eigen::VectorXf gradSmax = get<1>(finalRes);
  // std::cout << "[DEBUG] gradSmax sum: " << gradSmax.sum()
  //           << ", size: " << gradSmax.size() << std::endl;
  // std::cout << "[DEBUG] gradSmax: " << gradSmax.transpose() << std::endl;
  // Apply chain rule to get the gradient with respect to the original vertices
  Eigen::VectorXf finalGrad = Eigen::VectorXf::Zero(num_V);
  finalGrad = gradSmax.transpose() * jacobian;

  // 4. Assemble spatial gradients for P and R
  Eigen::MatrixXf grad_box1 = Eigen::MatrixXf::Zero(num_P, 3);
  Eigen::MatrixXf grad_box2 = Eigen::MatrixXf::Zero(num_R, 3);

  for (int i = 0; i < num_N; ++i) {
    float w_i = gradSmax(i);
    Eigen::Vector3f d_i = normalsSet[i].normalized();

    // For box1 vertices (P)
    for (int p_idx = 0; p_idx < num_P; ++p_idx) {
      float sum_v = 0.0f;
      int base = p_idx * num_R;
      for (int r_idx = 0; r_idx < num_R; ++r_idx) {
        sum_v += jacobian(i, base + r_idx);
      }
      grad_box1.row(p_idx) += w_i * sum_v * d_i.transpose();
    }
    // Eigen::Vector3f sum_grad_box1 = grad_box1.colwise().sum();
    // std::cout << "[DEBUG] sum of grad_box1 over vertices (should match "
    //              "translation gradient): "
    //           << sum_grad_box1.transpose() << std::endl;

    // For box2 vertices (R) – note the negative sign because v = p - r
    for (int r_idx = 0; r_idx < num_R; ++r_idx) {
      float sum_v = 0.0f;
      for (int p_idx = 0; p_idx < num_P; ++p_idx) {
        sum_v += jacobian(i, p_idx * num_R + r_idx);
      }
      grad_box2.row(r_idx) -= w_i * sum_v * d_i.transpose(); // minus sign!
    }
  }
  // Lambda to compute scalar distance given box1 vertices P_mod
  // auto computeDistance =
  //     [&](const std::vector<Eigen::Vector3f> &P_mod) -> float {
  //   std::vector<Eigen::Vector3f> minkowski_mod;
  //   minkowski_mod.reserve(num_P * num_R);
  //   for (const auto &p : P_mod)
  //     for (const auto &r : R)
  //       minkowski_mod.push_back(p - r);
  //
  //   std::vector<float> innerMins_mod;
  //   for (size_t i = 0; i < num_N; ++i) {
  //     Eigen::Vector3f d = normalsSet[i].normalized();
  //     std::vector<float> dots(num_V);
  //     for (size_t j = 0; j < num_V; ++j)
  //       dots[j] = d.dot(minkowski_mod[j]);
  //     float dist_mod = std::get<0>(smoothMinListWithGradient(dots, r));
  //     innerMins_mod.push_back(dist_mod);
  //   }
  //   return std::get<0>(smoothMaxListWithGradient(innerMins_mod, r));
  // };
  // // ----- DEBUG: Numerical per-vertex gradient check -----
  // float eps = 1e-4f;
  // Eigen::MatrixXf num_grad_box1(num_P, 3);
  // for (int v = 0; v < num_P; ++v) {
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
  // std::cout << "[DEBUG] Numerical grad_box1:\n" << num_grad_box1 << std::endl;
  // std::cout << "[DEBUG] Difference:\n"
  //           << (grad_box1 - num_grad_box1) << std::endl;
  // // --------------------------------------------------------
  // Eigen::Vector3f ana_grad_t = grad_box1.colwise().sum();
  // Eigen::Vector3f num_grad_t = num_grad_box1.colwise().sum();
  // std::cout << "[DEBUG] Analytical translation grad: " << ana_grad_t.transpose()
  //           << std::endl;
  // std::cout << "[DEBUG] Numerical  translation grad: " << num_grad_t.transpose()
  //           << std::endl;
  return make_tuple(finalDist, finalGrad, grad_box1, grad_box2);
}
