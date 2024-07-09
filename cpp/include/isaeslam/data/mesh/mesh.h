#ifndef MESH_H
#define MESH_H

#include <thread>
#include <vector>

#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/frame.h"
#include "isaeslam/data/landmarks/ALandmark.h"
#include "pcl/common/transforms.h"
#include "utilities/geometry.h"
#include "utilities/imgProcessing.h"
#include <pcl/io/pcd_io.h>

namespace isae {

typedef std::vector<std::shared_ptr<AFeature>> FeatPolygon;
typedef std::vector<std::shared_ptr<ALandmark>> LmkPolygon;

class Mesh3D;
struct Vertex;
struct Polygon;

class Mesh3D {
  public:
    Mesh3D() = default;
    Mesh3D(double ZNCC_tsh, double max_length_tsh) : _ZNCC_tsh(ZNCC_tsh), _max_length_tsh(max_length_tsh) {}
    ~Mesh3D() = default;

    std::vector<std::shared_ptr<Polygon>> getPolygonVector() const {
        std::lock_guard<std::mutex> lock(_mesh_mtx);
        return _polygons;
    }
    pcl::PointCloud<pcl::PointNormal> getPointCloud() const {
        std::lock_guard<std::mutex> lock(_pc_mtx);
        return _pcl_cloud;
    }
    std::shared_ptr<Frame> getFrame() const {
        std::lock_guard<std::mutex> lock(_mesh_mtx);
        return _reference_frame;
    }

    void updateMesh(std::vector<FeatPolygon> feats_polygon, std::shared_ptr<Frame> frame);
    std::unordered_map<std::shared_ptr<ALandmark>, std::shared_ptr<Vertex>> getMap() { return _map_lmk_vertex; }
    bool checkTriangle(std::vector<std::shared_ptr<Vertex>> vertices);
    bool checkPolygon(std::shared_ptr<Polygon> polygon);
    bool checkPolygonArea(std::shared_ptr<Polygon> polygon, double area2d);
    bool checkPolygonTri(std::shared_ptr<Polygon> polygon3d, FeatPolygon polygon2d);
    void removePolygon(std::shared_ptr<Polygon> polygon) {
        std::lock_guard<std::mutex> lock(_mesh_mtx);
        for (int i = _polygons.size() - 1; i >= 0; i--) {
            if (_polygons.at(i) == polygon) {
                _polygons.erase(_polygons.begin() + i);
            }
        }
    }
    void analysePolygon(std::shared_ptr<Polygon> polygon);
    void filterMesh();
    void projectMesh();
    void generatePointCloud();

    mutable std::mutex _mesh_mtx;
    mutable std::mutex _pc_mtx;

  private:
    std::unordered_map<std::shared_ptr<ALandmark>, std::shared_ptr<Vertex>> _map_lmk_vertex;
    std::vector<std::shared_ptr<Polygon>> _polygons;
    std::vector<Eigen::Vector3d> _point_cloud;
    pcl::PointCloud<pcl::PointNormal> _pcl_cloud;
    std::unordered_map<std::shared_ptr<Polygon>, std::vector<Eigen::Vector2d>> _map_poly_tri2d;

    // Storage of frame related objects
    std::shared_ptr<Frame> _reference_frame;
    std::shared_ptr<ImageSensor> _cam0, _cam1;
    cv::Mat _img0, _img1;
    Eigen::Affine3d _T_w_cam0;

    // Tuning parameters
    double _ZNCC_tsh       = 0.8;
    double _max_length_tsh = 5;
};

struct Vertex {
  public:
    Vertex() {}

    Vertex(std::shared_ptr<ALandmark> lmk) : _lmk(lmk) {}
    ~Vertex() = default;

    Eigen::Vector3d getVertexPosition() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _lmk->getPose().translation();
    }
    Eigen::Vector3d getVertexNormal() const { return _vertex_normal; }
    std::vector<std::shared_ptr<Polygon>> getPolygons() const { return _polygons; }
    std::shared_ptr<ALandmark> getLmk() const { return _lmk; }

    void addPolygon(std::shared_ptr<Polygon> polygon) { _polygons.push_back(polygon); }
    void removePolygon(std::shared_ptr<Polygon> polygon) {
        for (int i = _polygons.size() - 1; i >= 0; i--) {
            if (_polygons.at(i) == polygon) {
                _polygons.erase(_polygons.begin() + i);
            }
        }
    }

  private:
    std::shared_ptr<ALandmark> _lmk;
    Eigen::Vector3d _vertex_normal;
    std::vector<std::shared_ptr<Polygon>> _polygons;

    mutable std::mutex _lmk_mtx;
};

struct Polygon : std::enable_shared_from_this<Polygon> {
  public:
    Polygon() {}

    Polygon(std::vector<std::shared_ptr<Vertex>> vertices) : _vertices(vertices) { _outlier = false; }
    ~Polygon() = default;

    void setNormal(Eigen::Vector3d normal) { _normal = normal; }
    void setBarycenter(Eigen::Vector3d barycenter) { _barycenter = barycenter; }
    void setCovariance(Eigen::Matrix2d covariance) { _covariance = covariance; }
    void setScore(double score) { _traversability_score = score; }
    void setOutlier() { _outlier = true; }

    Eigen::Vector3d getPolygonNormal() const { return _normal; }
    Eigen::Vector3d getBarycenter() const { return _barycenter; }
    Eigen::Matrix2d getCovariance() const { return _covariance; }
    std::vector<std::shared_ptr<Vertex>> getVertices() const { return _vertices; }
    double getScore() const { return _traversability_score; }
    bool isOutlier() { return _outlier; }

  private:
    Eigen::Vector3d _normal;
    Eigen::Vector3d _barycenter;
    Eigen::Matrix2d _covariance;
    double _traversability_score;
    bool _outlier;

    std::vector<std::shared_ptr<Vertex>> _vertices;
};

} // namespace isae

#endif // MESH_H