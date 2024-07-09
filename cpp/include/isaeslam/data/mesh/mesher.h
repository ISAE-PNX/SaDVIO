#ifndef MESHER_H
#define MESHER_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <thread>

#include "utilities/timer.h"
#include "isaeslam/data/mesh/mesh.h"
#include "isaeslam/data/sensors/ASensor.h"
#include "isaeslam/featuredetectors/aFeatureDetector.h"

namespace isae {

class Mesher {

  public:
    Mesher(std::string slam_mode, double ZNCC_tsh, double max_length_tsh);

    std::vector<FeatPolygon> createMesh2D(std::shared_ptr<ImageSensor> sensor);
    std::vector<cv::Vec6f> computeMesh2D(const cv::Size img_size, const std::vector<cv::Point2f> p2f_to_triangulate);
    void addNewKF(std::shared_ptr<Frame> frame);
    bool getNewKf();

    void run();

    std::queue<std::shared_ptr<Frame>> _kf_queue;
    std::shared_ptr<Frame> _curr_kf;
    std::shared_ptr<Mesh3D> _mesh_3d;
    std::string _slam_mode;
    double _avg_mesh_t;
    int _n_kf;

    mutable std::mutex _mesher_mtx;
};

} // namespace isae

#endif // MESHER_H