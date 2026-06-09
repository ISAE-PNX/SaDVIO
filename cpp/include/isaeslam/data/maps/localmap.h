#ifndef LOCALMAP_H
#define LOCALMAP_H

#include "isaeslam/data/maps/amap.h"

namespace isae {

/*!
 * @brief Class for a Local Map
 *
 * This is a local map that implements a sliding window approach.
 */
class LocalMap : public AMap {
  public:
    LocalMap() = default;
    LocalMap(size_t min_kf_number, size_t max_kf_number, size_t fixedFrameNumber);

    /*!
     * @brief Add a frame to the local map.
     *
     * This method adds a frame to the local map and also pushes its landmarks into the map.
     * It implements a sliding window approach, removing the oldest frame if the maximum number of keyframes is reached.
     */
    void addFrame(std::shared_ptr<Frame> &frame) override;

    std::shared_ptr<isae::Frame> getLastFrame() override;

    typed_vec_landmarks &getLandmarks() override;

    /*!
     * @brief Remove a frame from the local map.
     *
     * This method removes a frame from the local map and also removes its landmarks from the map.
     * It updates the sliding window accordingly.
     */
    void removeFrame(std::shared_ptr<Frame> &frame);

    size_t getWindowSize() { return _max_kf_number; }
    size_t getFixedFrameNumber() { return _fixed_frames_number; }
    std::vector<Eigen::Affine3d> &getOldFramesPoses() { return _removed_frame_poses; }

    /*!
     * @brief Compute the relative pose between two frames.
     *
     * This method computes the relative pose between two frames of the local map
     * and derives the covariance of the pose estimation.
     */
    bool computeRelativePose(std::shared_ptr<Frame> &frame1,
                             std::shared_ptr<Frame> &frame2,
                             Eigen::Affine3d &T_f1_f2,
                             Eigen::MatrixXd &cov);

    bool getMarginalizationFlag() { 
      // If we have too much frames, raise the marginalization flag
      if (_frames.size() > _max_kf_number) {
          _margin_flag = true;
      } else {
          _margin_flag = false;
      } 
      return _margin_flag;
    }

    /*!
     * @brief Discard the last frame from the local map.
     *
     * This method discards the last frame from the local map and updates the sliding window accordingly.
     * It also sets the marginalization flag to false.
     */
    void discardLastFrame();

    /*!
     * @brief Reset the local map.
     *
     * This method resets the local map, clearing all frames and landmarks.
     * It is useful for reinitializing the local map in case of a failure or a new session.
     */
    void reset();

  protected:
    /*!
     * @brief Remove landmarks from the local map that do not have any features.
     *
     * This method iterates through all landmarks in the local map and removes those that have no associated features.
     * It is called after discarding a frame to ensure the map remains clean.
     */
    void removeEmptyLandmarks();

    size_t _min_kf_number       = 1; //!< number of keyframes that are added by default when the map starts
    size_t _max_kf_number       = 7; //!< size of the sliding window
    size_t _fixed_frames_number = 1; //!< number of frame that remain static during windowed BA
    bool _margin_flag;               //!< flag raised if the the last frame needs to be marginalized
    std::vector<Eigen::Affine3d> _removed_frame_poses; //!< old frames poses, for debugging purposes

    mutable std::mutex _localmap_mtx;
};

} // namespace isae

#endif // LOCALMAP_H