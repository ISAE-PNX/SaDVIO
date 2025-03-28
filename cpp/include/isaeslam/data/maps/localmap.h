#ifndef LOCALMAP_H
#define LOCALMAP_H

#include "isaeslam/data/maps/amap.h"

namespace isae {

class LocalMap : public AMap {
  public:
    LocalMap() = default;
    LocalMap(size_t min_kf_number, size_t max_kf_number, size_t fixedFrameNumber);
    void addFrame(std::shared_ptr<Frame> &frame) override;

    size_t getWindowSize() { return _max_kf_number; }
    size_t getFixedFrameNumber() { return _fixed_frames_number; }
    bool computeRelativePose(std::shared_ptr<Frame> &frame1,
                             std::shared_ptr<Frame> &frame2,
                             Eigen::Affine3d &T_f1_f2,
                             Eigen::MatrixXd &cov);
    bool getMarginalizationFlag() { return _margin_flag; }
    void discardLastFrame();
    void reset();

  protected:
    void pushLandmarks(std::shared_ptr<Frame> &frame) override;
    void removeEmptyLandmarks();

    // Parameters of the local map and KF selector (overwrote by value in config file)
    size_t _min_kf_number       = 1; // number of keyframes that are added by default when the map starts
    size_t _max_kf_number       = 7; // size of the sliding window
    size_t _fixed_frames_number = 1; // number of frame that remain static during windowed BA
    bool _margin_flag;               // flag raised if the the last frame needs to be marginalized

    mutable std::mutex _localmap_mtx;
};

} // namespace isae

#endif // LOCALMAP_H