#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "nanoflann.hpp"

using namespace std;
using namespace Eigen;

const float VERYBIGNUMBER = 10000000.0;
const float VERYSMALLNUMBER = 1/VERYBIGNUMBER;

// Global constants for the default delta and ds valuesAdd commentMore actions
extern double c_delta;
extern double c_ds;
// Approximate value for theta=0 in SE(3) EE distance
extern double c_maxCosTheta;
extern double c_theta_zero;

struct DistResult
{
    float D;
    float true_D;
    VectorXf grad_D;

    string toString() const;
};

struct PrimInfo
{
    float lx_d;
    float ly_d;
    float lz_d;
    float px, py, pz;
    float Qxx, Qxy, Qxz;
    float Qyx, Qyy, Qyz;
    float Qzx, Qzy, Qzz;
    int type;

    PrimInfo();
};

struct FKResult
{
    int no_links;
    vector<Matrix4f> htm_dh;
    vector<MatrixXf> jac_v_dh;
    vector<MatrixXf> jac_w_dh;
    Matrix4f htm_ee;
    MatrixXf jac_v_ee;
    MatrixXf jac_w_ee;

    FKResult(int _no_links);
    FKResult();

    string toString() const;

    Vector3f get_x_dh(int ind_link) const;
    Vector3f get_y_dh(int ind_link) const;
    Vector3f get_z_dh(int ind_link) const;
    Vector3f get_p_dh(int ind_link) const;
    Matrix3f get_Q_dh(int ind_link) const;

    Vector3f get_x_ee() const;
    Vector3f get_y_ee() const;
    Vector3f get_z_ee() const;
    Vector3f get_p_ee() const;
    Matrix3f get_Q_ee() const;
};

struct FKPrimResult
{

    vector<Matrix4f> htm_prim;
    vector<MatrixXf> jac_v_prim;
    vector<MatrixXf> jac_w_prim;

    FKPrimResult();

    string toString() const;
};

struct TaskResult
{
    VectorXf task;
    MatrixXf jac_task;
    float max_error_pos;
    float max_error_ori;

    TaskResult();

    string toString() const;
};

struct IKResult
{
    VectorXf qf;
    float error_pos;
    float error_ori;
    bool success;

    IKResult();

    string toString() const;
};

struct VectorFieldResult
{
    VectorXf twist;
    float dist;
    int index;
};

struct PrimDistResult
{
    float dist;
    Vector3f proj_A;
    Vector3f proj_B;
    Vector3f aux;

    //
    vector<float> hist_error;
    //

    string toString() const;
};

struct QueueElement
{
    int nodeIndex;
    float dist;
    Vector3f proj_A;
    Vector3f proj_B;

    bool operator>(const QueueElement& other) const
    {
        return dist > other.dist;
    }
};

struct ProjResult
{
    float dist;
    Vector3f proj;

    string toString() const;
};

struct AABB
{
    float lx;
    float ly;
    float lz;
    Vector3f p;

    AABB();
    static AABB get_aabb_pointcloud(const vector<Vector3f>& points, int start, int end);
    static float dist_aabb(AABB aabb1, AABB aabb2);
};


struct BVH
{
    vector<AABB> aabb;
    vector<int> parent;
    vector<int> left_child;
    vector<int> right_child;

    static int build_bvh(BVH &bvh, vector<Vector3f>& points, int start, int end, int parentIndex);
    BVH();
    BVH(vector<Vector3f>& points);
};

struct CheckFreeConfigResult
{
    bool isfree;
    string message;
    vector<int> info;
};

struct DistStructLinkObj
{
    bool is_null;
    int link_number;
    int link_col_obj_number;
    float distance;
    Vector3f point_link;
    Vector3f point_object;
    MatrixXf jac_distance;

    DistStructLinkObj();
};

struct DistStructRobotObj
{
    bool is_null;
    MatrixXf jac_dist_mat;
    VectorXf dist_vect;
    vector<DistStructLinkObj> list_info;

    DistStructRobotObj();

    DistStructLinkObj get_item(int ind_link, int ind_obj_link);
};


struct DistStructLinkLink
{
    bool is_null;
    int link_number_1;
    int link_number_2;
    int link_col_obj_number_1;
    int link_col_obj_number_2;
    float distance;
    Vector3f point_link_1;
    Vector3f point_link_2;
    MatrixXf jac_distance;

    DistStructLinkLink();
};

struct DistStructRobotAuto
{
    bool is_null;
    MatrixXf jac_dist_mat;
    VectorXf dist_vect;
    vector<DistStructLinkLink> list_info;

    DistStructRobotAuto();

    DistStructLinkLink get_item(int ind_link_1, int ind_link_2, int ind_obj_link_1, int ind_obj_link_2);
};


using PointCloud =  std::shared_ptr<nanoflann::PointCloud<float>>;

using KDTree = std::shared_ptr<nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, nanoflann::PointCloud<float>>,
        nanoflann::PointCloud<float>, 3 
        >>;



struct GeometricPrimitives
{
    float lx;
    float ly;
    float lz;
    Matrix4f htm;
    int type;

    //For point cloud
    KDTree kdtree; 
    PointCloud pointcloud;
    BVH bvh;

    //For convex polytopes
    MatrixXf A;
    VectorXf b;

    //For point cloud and convex polytopes
    vector<Vector3f> points_gp;
    Vector3f center;


    GeometricPrimitives();

    static GeometricPrimitives create_sphere(Matrix4f htm, float radius);
    static GeometricPrimitives create_box(Matrix4f htm, float width, float depth, float height);
    static GeometricPrimitives create_cylinder(Matrix4f htm, float radius, float height);
    static GeometricPrimitives create_pointcloud(vector<Vector3f> &points);
    static GeometricPrimitives create_convexpolytope(Matrix4f htm, MatrixXf A, VectorXf b);

  
    GeometricPrimitives to_pointcloud(float disc) const;
    ProjResult projection(Vector3f point, float h, float eps) const;
    Vector3f support(Vector3f direction) const;
    PrimDistResult dist_to(GeometricPrimitives prim, float h, float eps, float tol, int no_iter_max) const;
    PrimDistResult dist_to(GeometricPrimitives prim, float h, float eps, float tol, int no_iter_max, Vector3f p_A0) const;

    AABB get_aabb() const;
    GeometricPrimitives copy() const;

    string toString() const;
};

struct Manipulator
{
    int no_links;
    int no_tube_points;
    int no_prim;

    vector<float> theta;
    vector<float> dh_cos_theta;
    vector<float> dh_sin_theta;
    vector<float> dh_d;
    vector<float> alpha;
    vector<float> dh_cos_alpha;
    vector<float> dh_sin_alpha;
    vector<float> dh_a;
    vector<int> joint_type;

    Matrix4f htm_world_to_dh0;
    Matrix4f htm_dhn_to_ee;

    VectorXf q_min;
    VectorXf q_max;

    vector<vector<Vector3f>> coord_tube;
    vector<vector<GeometricPrimitives>> geo_prim;

  // Inertial parameters
    vector<float> masses;
    vector<Matrix3f> inertia_tensors;
    vector<Vector3f> com_positions;
    Vector3f gravity_vec = Vector3f(0, 0, -9.81);

    float tube_radius;

    // Constructors
    Manipulator(int _no_links);
    Manipulator();

    // String
    string toString() const;
    void set_joint_param(int ind_link, float _theta, float _d, float _alpha, float _a, int _joint_type, float _q_min, float _q_max);
    void set_inertial_param(int ind_link, float _mass, Vector3f _com, Matrix3f _inertia, bool wrt_dh=false);
    void add_tube_coord(int ind_link, Vector3f coord);
    void add_geo_prim(int ind_link, GeometricPrimitives prim);
    void set_htm_extra(Matrix4f _htm_world_to_dh0, Matrix4f _htm_dhn_to_ee);

    // Other functions

    vector<FKResult> fk(const vector<VectorXf> &q, const vector<Matrix4f> &htm_world_base, bool compute_jac = true) const;

    FKResult fk(VectorXf q, Matrix4f htm_world_base, bool compute_jac = true) const;

    vector<FKPrimResult> fk_prim(const vector<VectorXf> &q, const vector<FKResult> &fk_res_all) const;

    FKPrimResult fk_prim(VectorXf q, Matrix4f htm_world_base) const;

    TaskResult fk_task(VectorXf q, Matrix4f htm_world_base, Matrix4f tg_htm) const;

    IKResult ik(Matrix4f tg_htm, Matrix4f htm, VectorXf q0, float p_tol, float a_tol, int no_iter_max, bool ignore_orientation) const;

    CheckFreeConfigResult check_free_configuration(VectorXf q, Matrix4f htm, vector<GeometricPrimitives> obstacles, bool check_joint,
                                                   bool check_auto, float tol, float dist_tol, int no_iter_max) const;

    DistStructRobotObj compute_dist(GeometricPrimitives obj, VectorXf q, Matrix4f htm, DistStructRobotObj old_dist_struct,
                                                 float tol, int no_iter_max, float max_dist, float h, float eps) const;

    DistStructRobotAuto compute_dist_auto(VectorXf q, DistStructRobotAuto old_dist_struct,
                                                float tol, int no_iter_max, float max_dist, float h, float eps) const;

    std::tuple<MatrixXf, VectorXf, VectorXf> getDynamicModel(const VectorXf &_q, const VectorXf &_qp) const;
    VectorXf newtonEuler(const VectorXf &q, const VectorXf &qdot,
                      const VectorXf &qddot) const;

};

int mini(int a, int b);

float maxf(float a, float b);

float minf(float a, float b);

float shape_fun(float x, float cp, float cn, float eps);

float shape_fun_der(float x, float cp, float cn, float eps);

string print_number(float x, int nochar = 8);

string print_vector(VectorXf v, int nochar = 8);

string print_matrix(MatrixXf M, int nochar = 8);

Matrix3f s_mat(Vector3f v);

Matrix4f rotx(float theta);

Matrix4f roty(float theta);

Matrix4f rotz(float theta);

Matrix4f trn(float x, float y, float z);

float urand(float v_min, float v_max);

float urand();

VectorXf urand_vec(int n, float v_min, float v_max);

VectorXf urand_vec(VectorXf v_min, VectorXf v_max);

MatrixXf null_space(MatrixXf A);

MatrixXf m_vert_stack(MatrixXf A1, MatrixXf A2);

MatrixXf m_hor_stack(MatrixXf A1, MatrixXf A2);

VectorXf v_ver_stack(VectorXf v1, VectorXf v2);

VectorXf v_ver_stack(float v1, VectorXf v2);

VectorXf v_ver_stack(float v1, float v2);

VectorXf v_ver_stack(VectorXf v1, float v2);

vector<float> quadratic_interp(vector<float> x, int N);

vector<VectorXf> quadratic_interp(vector<VectorXf> x, int N);

vector<VectorXf> quadratic_interp_vec(vector<VectorXf> x, int N);

PrimDistResult dist_line(vector<Vector3f> a0, vector<Vector3f> a1, vector<Vector3f> b0, vector<Vector3f> b1);

VectorFieldResult vectorfield_rn(VectorXf q, vector<VectorXf> &q_path, float alpha, float const_velocity, bool is_closed, float gamma);

VectorXf dp_inv_solve(const MatrixXf& A, const VectorXf& b, float eps);

VectorXf sqrt_sign(VectorXf v);

vector<Vector3f> get_vertex(const MatrixXf& A, const VectorXf& b);

VectorXf solveQP(const MatrixXf& H,const VectorXf& f,const MatrixXf& A,const VectorXf& b,const MatrixXf& Aeq,const VectorXf& beq);

VectorXf solveQP(const MatrixXf& H,const VectorXf& f,const MatrixXf& A,const VectorXf& b);

VectorFieldResult vectorfield_SE3(const Eigen::Matrix4d& state, const vector<Eigen::Matrix4d>& curve, float kt1, float kt2, float kt3, float kn1, float kn2,
    const vector<Eigen::MatrixXd>& curve_derivative=std::vector<Eigen::MatrixXd>(), double delta=c_delta, double ds=c_ds);

Eigen::Matrix4d expSE3(const Eigen::Matrix4d X);

Eigen::Matrix4d SmapSE3(const Eigen::VectorXd xi);

Eigen::VectorXd invSmapSE3(const Eigen::Matrix4d A);

VectorXf recursiveNewtonEuler(const VectorXf &q, const VectorXf &qdot,
                      const VectorXf &qddot, const Vector3f &g0,
                      const Manipulator &robot);

std::tuple<MatrixXf, VectorXf, VectorXf>
getEulerLagrangeMatrices(const VectorXf &_q, const VectorXf &_qp,
                         const Vector3f &_g0, const Manipulator &robot);

