#ifndef SLAM_ISAE_MODEL3D_H
#define SLAM_ISAE_MODEL3D_H


#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>


namespace isae {

    class AModel3d : public std::enable_shared_from_this<AModel3d> {
    public:
        AModel3d(){}
        std::vector<Eigen::Vector3d> getModel(){return model;}

    protected:
        std::vector<Eigen::Vector3d> model;
    };


    //******************************************************************
    // point 3D
    class ModelPoint3D : public AModel3d {
    public:
        ModelPoint3D(){
            model.push_back(Eigen::Vector3d(0,0,0));
        }
    };

    //******************************************************************
    // edgelet 3D
    class ModelEdgelet3D : public AModel3d {
    public:
        ModelEdgelet3D(){
            model.push_back(Eigen::Vector3d(0,0,0));
            model.push_back(Eigen::Vector3d(1,0,0)); // to represent the orientation
        }
    };


    //******************************************************************
    // line 3D
    class ModelLine3D : public AModel3d {
    public:
        ModelLine3D(){
            model.push_back(Eigen::Vector3d(-0.5,0,0)); // start point
            model.push_back(Eigen::Vector3d(0.5,0,0));  // end point
        }
    };

    //******************************************************************
    // ellipse pattern 3D
    class ModelEllipsePattern3D : public AModel3d {
    public:
        ModelEllipsePattern3D(){
            model.push_back(Eigen::Vector3d(0,0,0));
            model.push_back(Eigen::Vector3d(0,1,0));
            model.push_back(Eigen::Vector3d(1,0,0));
            model.push_back(Eigen::Vector3d(1,1,0));
        }
    };


    //******************************************************************
    // Bounding Box 3D
    class ModelBBox3D : public AModel3d {
    public:
        ModelBBox3D(){
            model.push_back(Eigen::Vector3d(0,0,0));
            model.push_back(Eigen::Vector3d(0,1,0));
            model.push_back(Eigen::Vector3d(1,0,0));
            model.push_back(Eigen::Vector3d(1,1,0));
            model.push_back(Eigen::Vector3d(0,0,1));
            model.push_back(Eigen::Vector3d(0,1,1));
            model.push_back(Eigen::Vector3d(1,0,1));
            model.push_back(Eigen::Vector3d(1,1,1));
        }
    };


}
#endif //SLAM_ISAE_MODEL3D_H
