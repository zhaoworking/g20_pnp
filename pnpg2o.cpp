
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>
#include<fstream>
#include<iostream>
#include "pnpg2o.hpp"
#include <chrono>
#define SET_CLOCK(t0) \
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now()
#define TIME_DIFF(t1, t0) \
    (std::chrono::duration_cast<std::chrono::duration<double>>((t1) - (t0)).count())

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
    std::pair<cv::Point2f,cv::Point2f> pt2d12;
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
    std::vector<std::pair<cv::Point2f,cv::Point2f>> pts2d12_inliers;
    std::vector<std::pair<cv::Point2f,cv::Point2f>> pts2d12_outliers;
    // string write_path;
public:
    PnPOptmization() = default;
    // PnPOptmization(const string& path):write_path{path}
    bool WritePoints(string& path,vector<pair<cv::Point2f,cv::Point2f>> inliers,
                                        std::vector<pair<cv::Point2f,cv::Point2f>> outliers){
        ofstream fPts;
        fPts.open(path.c_str());
        if(fPts.good() == false){
            std::cout << "Open file failed" << endl;
            return false;
        }
        fPts << "#inliers,pt2d1,pt2d2" << endl;
        for(int i = 0; i < inliers.size(); i++){
            fPts << inliers[i].first.x << "," << inliers[i].first.y << "," <<
                    inliers[i].second.x << "," << inliers[i].second.y << endl;
        }
        fPts << "#outliers,pt2d1,pt2d2" << endl;
        for(int i = 0; i < outliers.size(); i++){
            fPts << outliers[i].first.x << "," << outliers[i].first.y << "," <<
                    outliers[i].second.x << "," << outliers[i].second.y << endl;
        }
        fPts.close();

    }
    int LoadPoints(string& path){

        ifstream fPts;
        fPts.open(path.c_str());
        if(fPts.good() == false){
            std::cout << "Open file failed" << endl;
            return 0;
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
                float data[7];
                int count = 0;
                while ((pos = s.find(',')) != string::npos) {
                    item = s.substr(0, pos);
                    data[count++] = stof(item);
                    s.erase(0, pos + 1);
                }
                item = s.substr(0, pos);
                data[6] = stof(item);

                pts3d.push_back(cv::Point3f(data[0],data[1],data[2]));
                pts2d2.push_back(cv::Point2f(data[3],data[4]));
                pts2d1.push_back(cv::Point2f(data[5],data[6]));
            }
        }

        return pts3d.size();
    }
    void poseEstimationPnP(double sigmasquare)
    {
        Camera* ref_ = new Camera;

        cv::Mat K = ( cv::Mat_<double> ( 3,3 ) <<
                ref_->fx_, 0, ref_->cx_,
                0, ref_->fy_, ref_->cy_,
                0,0,1
                );
        cv::Mat rvec, tvec, inliers;
        cv::solvePnPRansac ( pts3d, pts2d2, K, cv::Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
        int num_inliers_ = inliers.rows;
        cout<<"pnpRansac inliers/total size: "<< num_inliers_ << "/" << pts3d.size() <<endl;
        Eigen::Vector3d rvec_( rvec.at<double> ( 0,0 ), rvec.at<double> ( 1,0 ), rvec.at<double> ( 2,0 ));
        double rvecnorm = rvec_.norm();
        Eigen::AngleAxisd rotation_vector(rvecnorm,rvec_ / rvecnorm);
        Eigen::Matrix3d R_ = rotation_vector.matrix();
        // T_c_w_estimated_ = Sophus::SE3d (
        //                     Sophus::SO3d(R_),
        //                     Eigen::Vector3d ( tvec.at<double> ( 0,0 ), tvec.at<double> ( 1,0 ), tvec.at<double> ( 2,0 ) )
        //                 );

        T_c_w_estimated_ = Sophus::SE3d (
                            Sophus::SO3d(Eigen::Matrix3d::Identity()),
                            Eigen::Vector3d(0,0,0)
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
        // pose->setFixed(true);
        pose->setEstimate ( g2o::SE3Quat (
            T_c_w_estimated_.rotationMatrix() , T_c_w_estimated_.translation()
        ));
        optimizer.addVertex ( pose );

        // double sigmasquare = 9 // sigma = 3pixel
        double chi2_th = 5.991 ;  // robust kernel 阈值
        std::vector<EdgeProjectXYZ2UVPoseOnly*> edges;
        for ( int i=0; i<pts3d.size(); i++ )
        {
            int index = i;
            // 3D -> 2D projection
            EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
            edge->setId ( i );
            edge->setVertex ( 0, pose );
            edge->camera_ = ref_;
            edge->point_ = Eigen::Vector3d ( pts3d[index].x, pts3d[index].y, pts3d[index].z );
            edge->pt2d12 = {pts2d1[index],pts2d2[index]};
            edge->setMeasurement ( Eigen::Vector2d ( pts2d2[index].x, pts2d2[index].y ) );
            edge->setInformation ( Eigen::Matrix2d::Identity() );
            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(chi2_th);
            edge->setRobustKernel(rk);
            edges.push_back(edge);
            optimizer.addEdge ( edge );
            // set the inlier map points 
            // match_3dpts_[index]->matched_times_++;
        }
        optimizer.initializeOptimization();
        optimizer.optimize ( 100 );

        int cnt_outlier = 0, cnt_inlier = 0;
        int iteration = 0;
        // while (iteration < 20) {
            cnt_outlier = 0;
            cnt_inlier = 0;
            pts2d12_outliers.clear();
            pts2d12_inliers.clear();
            // determine if we want to adjust the outlier threshold
            for (auto &ef : edges) {
                if (ef->chi2() > chi2_th * sigmasquare) {
                    cnt_outlier++;
                    pts2d12_outliers.push_back(ef->pt2d12);
                } else {
                    cnt_inlier++;
                    pts2d12_inliers.push_back(ef->pt2d12);
                }
            }
             
            double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
            // cout << "inlier_ratio is: " << inlier_ratio << endl;
            // if (inlier_ratio > 0.6) {
                cout << "pnpG2O inlier/total size: " << cnt_inlier << "/" << double(cnt_inlier + cnt_outlier) << endl;
                cout << "pnpG2O inlier ratio: "  << inlier_ratio << endl;
                cout << "chi2_th is: " << chi2_th * sigmasquare << endl;
                // break;
            // } else {
            //     chi2_th *= 2;
            //     iteration++;
            // }
        // }

        T_c_w_estimated_ = Sophus::SE3d (
            pose->estimate().rotation(),
            pose->estimate().translation()
        );
        
        cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;

    }

    void PnPcheck(Eigen::Matrix4d pose_gt, double sigma)
    {
        vector<cv::Point2f> pts2d2_est;
        Camera* ref_ = new Camera;
        vector<pair<cv::Point2f,cv::Point2f>> pts2d12;
        // cv::Mat K = ( cv::Mat_<double> ( 3,3 ) <<
        //         ref_->fx_, 0, ref_->cx_,
        //         0, ref_->fy_, ref_->cy_,
        //         0,0,1
        //         );

        // T_c_w_estimated_ = pose_gt;

        for(int i = 0; i < pts3d.size();i++){
            Eigen::Matrix<double ,4,1> Point;
            Point(0,0) = pts3d[i].x;
            Point(1,0) = pts3d[i].y;
            Point(2,0) = pts3d[i].z;
            Point(3,0) = 1.0;
            Eigen::Vector3d Point_c2 = (pose_gt * Point).block<3,1>(0,0);
            double invz = 1 / Point_c2(2);
            cv::Point2f pt2d_est;
            pt2d_est.x = ref_->fx_ * Point_c2(0) * invz + ref_->cx_;
            pt2d_est.y = ref_->fy_ * Point_c2(1) * invz + ref_->cy_;

            pts2d2_est.push_back(pt2d_est); 
            pts2d12.push_back({pts2d1[i],pts2d2[i]});           
        }

        // double sigmasquare = 9 // sigma = 3pixel
        double chi2_th = 5.991 ;  // robust kernel 阈值
      
        int cnt_outlier = 0, cnt_inlier = 0;
        int iteration = 0;
        cnt_outlier = 0;
        cnt_inlier = 0;
        pts2d12_outliers.clear();
        pts2d12_inliers.clear();
        // determine if we want to adjust the outlier threshold
        for(int i = 0; i < pts2d2_est.size(); i++){
            float error1 = (pts2d2_est[i].x - pts2d2[i].x) ;
            float error2 = (pts2d2_est[i].y - pts2d2[i].y) ;
            float error = sqrt(error1 * error1 + error2* error2);
            
            if ((error / sigma) > chi2_th ) {
                cnt_outlier++;
                pts2d12_outliers.push_back(pts2d12[i]);
            } else {
                cnt_inlier++;
                pts2d12_inliers.push_back(pts2d12[i]);
            }
        }
             
            double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
            cout << "pnpG2O inlier/total size: " << cnt_inlier << "/" << double(cnt_inlier + cnt_outlier) << endl;
            cout << "pnpG2O inlier ratio: "  << inlier_ratio << endl;
            cout << "chi2_th is: " << chi2_th * sigma << endl;
        
            // cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;

    }
};

std::vector<Eigen::Matrix4d> get_pose(){
    Eigen::Matrix4d Tc2c1_vio1;
    Eigen::Matrix4d Tc2c1_vio2;
    Eigen::Matrix4d Tc2c1_vio3;
    Eigen::Matrix4d Tc2c1_vio4;
    Eigen::Matrix4d Tc2c1_vio5;
    Eigen::Matrix4d Tc2c1_vio6;
    Eigen::Matrix4d Tc2c1_vio7;
    Eigen::Matrix4d Tc2c1_vio8;
    Tc2c1_vio1 <<   7.80299991e-01 , 2.40184526e-01 ,-5.77458572e-01 , 5.12247467e+00,
                    -2.93159293e-01,  9.56060965e-01,  1.53007328e-03,  2.03693117e+00,
                    5.52453021e-01,  1.68099640e-01,  8.16410317e-01, -3.05874643e+00,
                    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00;
// 0.751117192237285,-0.265073083892572,0.604602388178137,4.91408048139128,
// 0.659873023117633,0.328879036651407,-0.675589682394274,-3.12967731200298,
// -0.0197623048419328,0.90640800335872,0.421940698964189,-1.46535675709235,
// 0,0,0,1;
    Tc2c1_vio2 <<    0.99922088,  0.0333519 , 0.02109068 ,-0.83449443,
                    -0.031668 ,   0.99662947, -0.07567266,  3.90671074,
                    -0.0235435,   0.07494627,  0.99691009, -6.93800119,
                    0.       ,   0.        ,  0.        ,  1.        ;

    Tc2c1_vio3 <<     0.93940607,  0.16774171, -0.29897948,  7.29281244,
                    -0.19233073,  0.97980963, -0.05458686,  1.53645385,
                    0.28378622,  0.10878607,  0.95269274, -3.07080691,
                    0.        ,  0.        ,  0.        ,  1.        ;
    Tc2c1_vio4 <<     0.99618853, -0.07539977,  0.04383232,  0.50471084,
                    0.07810449,  0.99491174, -0.06364949,  3.58514192,
                    -0.03880981,  0.06682923,  0.99700989, -7.18814604,
                    0.          ,0.        ,  0.        ,  1.        ;
    Tc2c1_vio5 <<   -0.91076612,  0.21657846, -0.35155026,  2.4072427 ,
                    -0.13220476,  0.65362671,  0.74518088, -3.06568188,
                    0.39117193 , 0.72516917, -0.56667887,  6.83596287,
                    0.         , 0.         , 0.        ,  1.           ;
    Tc2c1_vio6 <<    8.40646114e-01,  2.15842109e-01, -4.96729631e-01,  6.25246946e+00,
                    -2.49581083e-01,  9.68351454e-01, -1.59838581e-03,  2.05777642e+00,
                    4.80663099e-01,  1.25323061e-01,  8.67897042e-01, -3.58456648e+00,
                    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00;

    Tc2c1_vio7 <<    0.75344769,  0.37438292, -0.54052652,  6.18375297,
                    -0.35402998,  0.92371658,  0.14631446,  2.46531793,
                    0.55407052,  0.08113093,  0.8284994 , -4.63784513,
                    0.        ,  0.        ,  0.        ,  1.        ;
    Tc2c1_vio8 <<   -0.50635873,  0.34534419, -0.7901423 ,  5.19163521,
                    -0.46939242,  0.65826507,  0.58852452, -3.00064715,
                    0.72336604,  0.66890163, -0.17123756,  6.01130413,
                    0.        ,  0.        ,  0.        ,  1.        ;
            
    return vector<Eigen::Matrix4d>{Tc2c1_vio1,Tc2c1_vio2,Tc2c1_vio3,Tc2c1_vio4,Tc2c1_vio5,Tc2c1_vio6,
                                    Tc2c1_vio7,Tc2c1_vio8};

}


int main (int argc, char * argv[]) {

    vector<double> vSigmasquare{1.,2.,3.,4.,20.};
    std::vector<Eigen::Matrix4d> vPose_gt = get_pose();

    for(int k = 4; k < vSigmasquare.size();k++){
        double sigmasquare_ = vSigmasquare[k];
        for(int i = 1; i < 13;i++){
            cout << to_string(i) << ":" << endl;
            string i_str = to_string(i);
            string base_path = "/home/ztq/Desktop/test_postgraduate3/image_match/result/Points" + i_str;
            // string base_path = "/home/linux/Desktop/test_postgraduate3/image_match/result/Points7";
            string path = base_path + ".txt";
            string path_write = base_path + "_filtered" + ".txt";
        
            Eigen::Matrix4d Tc2c1_gt = vPose_gt[i-1];
            PnPOptmization pnp = PnPOptmization();
            int pts_size = pnp.LoadPoints(path);
            if(pts_size == 0) 
                continue;
            SET_CLOCK(t0);
            pnp.poseEstimationPnP(sigmasquare_);
            // pnp.PnPcheck(Tc2c1_gt,sigmasquare_);
            SET_CLOCK(t1);
            cout << "optimization time: " << TIME_DIFF(t1,t0) << endl;
            pnp.WritePoints(path_write,pnp.pts2d12_inliers,pnp.pts2d12_outliers);
            cout << "Writing Finished " << endl;
        }
    }


}
