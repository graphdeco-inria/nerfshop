#include <neural-graphics-primitives/editing/tools/progressive_hulls.h>

#include <igl/circulation.h>
#include <igl/linprog.h>
#include <igl/copyleft/quadprog.h>
#include <igl/decimate.h>
#include <igl/unique.h>
#include <igl/max_faces_stopping_condition.h>

NGP_NAMESPACE_BEGIN

using namespace Eigen;

inline double area(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
    return (v1-v0).cross(v2-v0).norm()/2.;
}

inline double perimeter(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
    return (v1 - v0).norm() + (v2 - v1).norm() + (v0 - v2).norm();
}

bool test_valence (const std::vector<int>& N_origin, const int max_valence) {
    return N_origin.size() - 2 <= max_valence;
}

bool test_compactness (
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    // Facets before collapse
    const std::vector<int>& N_origin, 
    const Eigen::Vector3d new_pos,
    const int v_collapse0, 
    const int v_collapse1,
    const double compactness_threshold) {
    
    double c_e = std::numeric_limits<double>::infinity();
    // std::cout << "New" << std::endl;
    for (auto t_origin: N_origin) {
        double t_area = area(V.row(F(t_origin, 0)), V.row(F(t_origin, 1)), V.row(F(t_origin, 2)));
        double t_perim = perimeter(V.row(F(t_origin, 0)), V.row(F(t_origin, 1)), V.row(F(t_origin, 2)));
        double c_area = 4*M_PI*t_area/(t_perim*t_perim);
        if (c_area > 1e-9) {
            c_e = std::min(c_e, c_area);
        }
    }
    // std::cout << c_e << std::endl;

    double c_v = std::numeric_limits<double>::infinity();
    for (auto t_origin: N_origin) {
        std::vector<Eigen::Vector3d> t_vertices = {V.row(F(t_origin, 0)), V.row(F(t_origin, 1)), V.row(F(t_origin, 2))};
        for (int j = 0; j < 3; j++) {
            if (F(t_origin, j) == v_collapse0 || F(t_origin, j) == v_collapse1) {
                t_vertices[j] = new_pos;
            }
        }   
        double t_area = area(t_vertices[0], t_vertices[1], t_vertices[2]);
        double t_perim = perimeter(t_vertices[0], t_vertices[1], t_vertices[2]);
        double c_area = 4*M_PI*t_area/(t_perim*t_perim);
        if (c_area > 1e-9) {
            c_v = std::min(c_v, c_area);
        }
    }
    // std::cout << c_v << std::endl;
    // std::cout << c_v / (c_e + 1e-9) << std::endl;
    return c_v / (c_e + 1e-9) > compactness_threshold;
}


void progressive_hulls_linear_cost_and_placement(
    const int e,
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const Eigen::MatrixXi & E,
    const Eigen::VectorXi & EMAP,
    const Eigen::MatrixXi & EF,
    const Eigen::MatrixXi & EI,
    double & cost,
    Eigen::RowVectorXd & p,
    const ProgressiveHullsParams& params) {
    
    assert(V.cols() == 3 && "V.cols() should be 3");
    // Gather list of unique face neighbors
    std::vector<int> Nall =  igl::circulation(e, true,EMAP,EF,EI);
    std::vector<int> Nother= igl::circulation(e,false,EMAP,EF,EI);
    Nall.insert(Nall.end(),Nother.begin(),Nother.end());
    std::vector<int> N;
    igl::unique(Nall, N);

    Eigen::MatrixXd A(N.size(),3);
    Eigen::VectorXd D(N.size());
    Eigen::VectorXd B(N.size());
    //cout<<"N=[";
    for(int i = 0;i<N.size();i++)
    {
        const int f = N[i];
        //cout<<(f+1)<<" ";
        const Eigen::RowVector3d & v01 = V.row(F(f,1))-V.row(F(f,0));
        const Eigen::RowVector3d & v20 = V.row(F(f,2))-V.row(F(f,0));
        A.row(i) = v01.cross(v20);
        B(i) = V.row(F(f,0)).dot(A.row(i));
        D(i) = 
        (Eigen::Matrix3d()<< V.row(F(f,0)), V.row(F(f,1)), V.row(F(f,2)))
        .finished().determinant();
    }

    Eigen::Vector3d f = A.colwise().sum().transpose();
    Eigen::VectorXd x;
    bool success = igl::linprog(f,-A,-B,MatrixXd(0,A.cols()),VectorXd(0,1),x);

    p = x.transpose();

    success &= !params.compactness_test || test_compactness(V, F, N, p, E(e, 0), E(e, 1), params.compactness_threshold);
    success &= !params.valence_test || test_valence(N, params.max_valence);

    if(success)
    {
        cost  = (1./6.)*(x.dot(f) - D.sum());
    } else {
        cost = std::numeric_limits<double>::infinity();
        p = Eigen::RowVectorXd::Constant(1,3,std::nan("inf-cost"));
    }
}

void progressive_hulls_quadratic_cost_and_placement(
  const int e,
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  const Eigen::MatrixXi & E,
  const Eigen::VectorXi & EMAP,
  const Eigen::MatrixXi & EF,
  const Eigen::MatrixXi & EI,
  double & cost,
  Eigen::RowVectorXd & p,
  // Controls the amount of quadratic energy to add (too small will introduce
    // instabilities and flaps)
  const ProgressiveHullsParams& params)
{
    using namespace Eigen;
    using namespace std;

    assert(V.cols() == 3 && "V.cols() should be 3");
    // Gather list of unique face neighbors
    vector<int> Nall =  igl::circulation(e, true,EMAP,EF,EI);
    vector<int> Nother= igl::circulation(e,false,EMAP,EF,EI);
    Nall.insert(Nall.end(),Nother.begin(),Nother.end());
    vector<int> N;
    igl::unique(Nall,N);
    // Gather:
    //   A  #N by 3 normals scaled by area,
    //   D  #N determinants of matrix formed by points as columns
    //   B  #N point on plane dot normal
    MatrixXd A(N.size(),3);
    VectorXd D(N.size());
    VectorXd B(N.size());

    for(int i = 0;i<N.size();i++)
    {
        const int f = N[i];
        const RowVector3d & v01 = V.row(F(f,1))-V.row(F(f,0));
        const RowVector3d & v20 = V.row(F(f,2))-V.row(F(f,0));
        A.row(i) = v01.cross(v20);
        B(i) = V.row(F(f,0)).dot(A.row(i));
        D(i) = 
        (Matrix3d()<< V.row(F(f,0)), V.row(F(f,1)), V.row(F(f,2)))
        .finished().determinant();
    }

    Vector3d f = A.colwise().sum().transpose();
    VectorXd x;

    bool success = false;
    {
        RowVectorXd mid = 0.5*(V.row(E(e,0))+V.row(E(e,1)));
        MatrixXd G =  params.w*Matrix3d::Identity(3,3);
        VectorXd g0 = (1.-params.w)*f - params.w*mid.transpose();
        const int n = A.cols();
        success = igl::copyleft::quadprog(
            G,g0,
            MatrixXd(n,0),VectorXd(0,1),
            A.transpose(),-B,x);
        cost  = (1.-params.w)*(1./6.)*(x.dot(f) - D.sum()) + 
        params.w*(x.transpose()-mid).squaredNorm() +
        params.w*(V.row(E(e,0))-V.row(E(e,1))).norm();
    }

    p = x.transpose();

  // A x >= B
  // A x - B >=0
  // This is annoyingly necessary. Seems the solver is letting some garbage
  // slip by.
    success = success && ((A*x-B).minCoeff()>-1e-10);
    success &= !params.compactness_test || test_compactness(V, F, N, p, E(e, 0), E(e, 1), params.compactness_threshold);
    success &= !params.valence_test || test_valence(N, params.max_valence);
    if(success)
    {
        //assert(cost>=0 && "Cost should be positive");
    }else
    {
        cost = std::numeric_limits<double>::infinity();
        //VectorXi NM;
        //igl::list_to_matrix(N,NM);
        //cout<<matlab_format((NM.array()+1).eval(),"N")<<endl;
        //cout<<matlab_format(f,"f")<<endl;
        //cout<<matlab_format(A,"A")<<endl;
        //cout<<matlab_format(B,"B")<<endl;
        //exit(-1);
        p = RowVectorXd::Constant(1,3,std::nan("inf-cost"));
    }
}

bool progressive_hulls_linear(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const size_t max_m,
    Eigen::MatrixXd & U,
    Eigen::MatrixXi & G,
    Eigen::VectorXi & J,
    const ProgressiveHullsParams& params) {
    
    int m = F.rows();
    Eigen::VectorXi I;
    auto cost_placement_f = [&, params] 
    (const int e,
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const Eigen::MatrixXi & E,
    const Eigen::VectorXi & EMAP,
    const Eigen::MatrixXi & EF,
    const Eigen::MatrixXi & EI,
    double & cost,
    Eigen::RowVectorXd & p)
    {
        progressive_hulls_linear_cost_and_placement(e, V, F, E, EMAP, EF, EI, cost, p, params);
    };
    return igl::decimate(
        V,
        F,
        cost_placement_f,
        igl::max_faces_stopping_condition(m,(const int)m,max_m),
        U,
        G,
        J,
        I);
}


bool progressive_hulls_quadratic(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const size_t max_m,
    Eigen::MatrixXd & U,
    Eigen::MatrixXi & G,
    Eigen::VectorXi & J,
    const ProgressiveHullsParams& params) {
    
    int m = F.rows();
    Eigen::VectorXi I;
    auto cost_placement_f = [&, params] 
    (const int e,
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const Eigen::MatrixXi & E,
    const Eigen::VectorXi & EMAP,
    const Eigen::MatrixXi & EF,
    const Eigen::MatrixXi & EI,
    double & cost,
    Eigen::RowVectorXd & p)
    {
        progressive_hulls_quadratic_cost_and_placement(e, V, F, E, EMAP, EF, EI, cost, p, params);
    };
    {
        std::ofstream outfile("test.obj");
        for (int i = 0; i < V.rows(); i++)
        {
            auto v = V.row(i);
            outfile << "v " <<  v.x() << " " << v.y() << " " << v.z() << std::endl;
        }
        for (int i = 0; i < F.rows(); i++)
        {
            auto f = F.row(i);
            outfile << "f " << f.x()+1 << " " << f.y()+1 << " " << f.z()+1 << std::endl;
        }
    }
    return igl::decimate(
        V,
        F,
        cost_placement_f,
        igl::max_faces_stopping_condition(m,(const int)m,max_m),
        U,
        G,
        J,
        I);
}


NGP_NAMESPACE_END
