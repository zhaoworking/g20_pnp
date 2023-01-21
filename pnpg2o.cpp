
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>
#include<fstream>
#include<iostream>
#include "pnpg2o.hpp"

using namespace std;
// cv::Point3f getPositionCV() {
//     return cv::Point3f( pos_(0,0), pos_(1,0), pos_(2,0) );
// }
class Camera
{
public:
    typedef std::shared_ptr<Camera> Ptr;
    Camera(){}
    float fx_ = 1179.645;
    float fy_ = 1179.645;
    float cx_ = 506.7878;
    float cy_ = 522.2906;
    float depth_scale_ = 0;

    // coordinate transform: world, camera, pixel
    Eigen::Vector2d camera2pixel ( const Eigen::Vector3d& p_c )
    {
        return Eigen::Vector2d (
                fx_ * p_c ( 0,0 ) / p_c ( 2,0 ) + cx_,
                fy_ * p_c ( 1,0 ) / p_c ( 2,0 ) + cy_
            );
    }

};
class EdgeProjectXYZ2UVPoseOnly: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap >
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    
    virtual bool read( std::istream& in ){}
    virtual bool write(std::ostream& os) const {}
    
    Eigen::Vector3d point_;
    Camera* camera_;
    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
        _error = _measurement - camera_->camera2pixel ( 
            pose->estimate().map(point_) );
    }

    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*> ( _vertices[0] );
        g2o::SE3Quat T ( pose->estimate() );
        Eigen::Vector3d xyz_trans = T.map ( point_ );
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];
        double z_2 = z*z;

        _jacobianOplusXi ( 0,0 ) =  x*y/z_2 *camera_->fx_;
        _jacobianOplusXi ( 0,1 ) = - ( 1+ ( x*x/z_2 ) ) *camera_->fx_;
        _jacobianOplusXi ( 0,2 ) = y/z * camera_->fx_;
        _jacobianOplusXi ( 0,3 ) = -1./z * camera_->fx_;
        _jacobianOplusXi ( 0,4 ) = 0;
        _jacobianOplusXi ( 0,5 ) = x/z_2 * camera_->fx_;

        _jacobianOplusXi ( 1,0 ) = ( 1+y*y/z_2 ) *camera_->fy_;
        _jacobianOplusXi ( 1,1 ) = -x*y/z_2 *camera_->fy_;
        _jacobianOplusXi ( 1,2 ) = -x/z *camera_->fy_;
        _jacobianOplusXi ( 1,3 ) = 0;
        _jacobianOplusXi ( 1,4 ) = -1./z *camera_->fy_;
        _jacobianOplusXi ( 1,5 ) = y/z_2 *camera_->fy_;
    }
};
class EdgeProjectXYZ2UVPoseOnly;

class PnPOptmization{

public:
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d1;
    std::vector<cv::Point2f> pts2d2;
    Sophus::SE3d T_c_w_estimated_;
 
public:
    PnPOptmization() = default;

    void LoadPoints(string& path){

        ifstream fPts;
        fPts.open(path.c_str());
        if(fPts.good() == false){
            std::cout << "Open file failed" << endl;
            return;
        }
        while(!fPts.eof())
        {
            string s;
            getline(fPts,s);
            if(s[0] == '#')
                continue;
            // 只有在当前行不为空的时候执行
            if(!s.empty())
            {
                string item;
                size_t pos = 0;
                float data[5];
                int count = 0;
                while ((pos = s.find(',')) != string::npos) {
                    item = s.substr(0, pos);
                    data[count++] = stof(item);
                    s.erase(0, pos + 1);
                }
                item = s.substr(0, pos);
                data[4] = stof(item);

                pts3d.push_back(cv::Point3f(data[0],data[1],data[2]));
                pts2d2.push_back(cv::Point2f(data[3],data[4]));
            }
        }
    }
    void poseEstimationPnP()
    {
        // construct the 3d 2d observations
        // vector<cv::Point3f> pts3d;
        // vector<cv::Point2f> pts2d;

        // for ( int index:match_2dkp_index_ )
        // {
        //     pts2d.push_back ( keypoints_curr_[index].pt );
        // }
        // for ( MapPoint::Ptr pt:match_3dpts_ )
        // {
        //     pts3d.push_back( pt->getPositionCV() );
        // }
        Camera* ref_ = new Camera;

        cv::Mat K = ( cv::Mat_<double> ( 3,3 ) <<
                ref_->fx_, 0, ref_->cx_,
                0, ref_->fy_, ref_->cy_,
                0,0,1
                );
        cv::Mat rvec, tvec, inliers;
        cv::solvePnPRansac ( pts3d, pts2d2, K, cv::Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
        int num_inliers_ = inliers.rows;
        cout<<"pnp inliers: "<<num_inliers_<<endl;
        Eigen::Vector3d rvec_( rvec.at<double> ( 0,0 ), rvec.at<double> ( 1,0 ), rvec.at<double> ( 2,0 ));
        double rvecnorm = rvec_.norm();
        Eigen::AngleAxisd rotation_vector(rvecnorm,rvec_ / rvecnorm);
        Eigen::Matrix3d R_ = rotation_vector.matrix();
        T_c_w_estimated_ = Sophus::SE3d (
                            Sophus::SO3d(R_),
                            Eigen::Vector3d ( tvec.at<double> ( 0,0 ), tvec.at<double> ( 1,0 ), tvec.at<double> ( 2,0 ) )
                        );

        // using bundle adjustment to optimize the pose
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
        Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
        Block* solver_ptr = new Block ( std::unique_ptr<Block::LinearSolverType>(linearSolver) );
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( std::unique_ptr<Block>(solver_ptr));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm ( solver );

        g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
        pose->setId ( 0 );
        pose->setEstimate ( g2o::SE3Quat (
            T_c_w_estimated_.rotationMatrix() , T_c_w_estimated_.translation()
        ));
        optimizer.addVertex ( pose );

        // edges
        for ( int i=0; i<inliers.rows; i++ )
        {
            int index = inliers.at<int> ( i,0 );
            // 3D -> 2D projection
            EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
            edge->setId ( i );
            edge->setVertex ( 0, pose );
            edge->camera_ = ref_;
            edge->point_ = Eigen::Vector3d ( pts3d[index].x, pts3d[index].y, pts3d[index].z );
            edge->setMeasurement ( Eigen::Vector2d ( pts2d2[index].x, pts2d2[index].y ) );
            edge->setInformation ( Eigen::Matrix2d::Identity() );
            optimizer.addEdge ( edge );
            // set the inlier map points 
            // match_3dpts_[index]->matched_times_++;
        }

        optimizer.initializeOptimization();
        optimizer.optimize ( 10 );

        T_c_w_estimated_ = Sophus::SE3d (
            pose->estimate().rotation(),
            pose->estimate().translation()
        );
        
        cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
    }
};

int main (int argc, char * argv[]) {
    string path = "/home/caichicken/helloworld/tqzhao/data/0000Points.txt";
    PnPOptmization pnp = PnPOptmization();
    pnp.LoadPoints(path);
    pnp.poseEstimationPnP();
}
