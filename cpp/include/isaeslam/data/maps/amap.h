#ifndef AMAP_H
#define AMAP_H

#include <deque>

#include "isaeslam/data/frame.h"
#include "isaeslam/typedefs.h"

namespace isae {

/*!
 * @brief Abstract class for a Map
 *
 * It contains a set of frame and landmarks. That can be accessed and modified by the SLAM system.
 */
class AMap {
  public:
    AMap() = default;

    /**
     *  @brief Add a frame to the map.
     *
     * This method is pure virtual and must be implemented by derived classes.
     * Indeed addition of a new frame may require other operations e.g. for a sliding window approach
     */
    virtual void addFrame(std::shared_ptr<isae::Frame> &frame) = 0;

    std::deque<std::shared_ptr<Frame>> &getFrames() { return _frames; }

    /**
     *  @brief Get the last (newest) frame added to the map.
     */
    std::shared_ptr<isae::Frame> getLastFrame() {
        if (_frames.empty())
            return nullptr;
        return _frames.back();
    }

    /**
     *  @brief Provides the last N frames added to the map.
     */
    void getLastNFramesIn(size_t N, std::vector<std::shared_ptr<isae::Frame>> &dest) {
        for (uint i = 0; i < std::min(N, _frames.size()); ++i) {
            dest.push_back(_frames.at(_frames.size() - 1 - i));
        }
    }

    typed_vec_landmarks &getLandmarks() { return _landmarks; }
    size_t getMapSize() { return _frames.size(); }

    /**
     *  @brief Add all the landmarks of a frame to the map.
     */
    void pushLandmarks(std::shared_ptr<isae::Frame> &frame) {
        typed_vec_landmarks all_ldmks = frame->getLandmarks();

        // For all type of landmarks to add
        for (auto &typed_ldmks : all_ldmks) {
            for (auto &ldmk : typed_ldmks.second) {
                if (!(!ldmk->isInitialized() || ldmk->isInMap() || ldmk->getFeatures().empty())) {
                    {
                        ldmk->setInMap();
                        _landmarks[ldmk->_label].push_back(ldmk);
                    }
                }
            }
        }
    }

  protected:
    std::deque<std::shared_ptr<Frame>> _frames; //!< A deque of frames in the map, ordered by time
    typed_vec_landmarks _landmarks;             //!< All types of landmark in the map stored as std vectors
};

} // namespace isae

#endif // AMAP_H