#ifndef LINE3DLANDMARKINITIALIZER_H
#define LINE3DLANDMARKINITIALIZER_H

#include "isaeslam/data/landmarks/Line3D.h"
#include "isaeslam/landmarkinitializer/alandmarkinitializer.h"
#include "isaeslam/typedefs.h"

namespace isae {

class Line3DLandmarkInitializer : public ALandmarkInitializer {
  public:
    Line3DLandmarkInitializer(uint nb_requiered_ldmk)
        : ALandmarkInitializer(nb_requiered_ldmk) {}

  protected:
    bool initLandmark(std::vector<std::shared_ptr<AFeature>> features, std::shared_ptr<ALandmark> &landmark) override;
    bool initLandmarkWithDepth(std::vector<std::shared_ptr<AFeature>> features,
                               std::shared_ptr<ALandmark> &landmark) override;

    // must link f or f1 & f2 to the landmark
    std::shared_ptr<ALandmark> createNewLandmark(std::shared_ptr<AFeature> f) override;

    static Eigen::Vector3d processOrientation(std::vector<Eigen::Vector3d> Ns);
    static Eigen::Vector3d processPosition(std::vector<Eigen::Vector3d> Ns, std::vector<Eigen::Vector3d> Os);
    static void processSegmentPoints(Eigen::Vector3d position,
                                     Eigen::Vector3d direction,
                                     std::vector<Eigen::Vector3d> Os,
                                     std::vector<Eigen::Vector3d> rays_s,
                                     std::vector<Eigen::Vector3d> rays_e,
                                     Eigen::Vector3d &start,
                                     Eigen::Vector3d &end);
};

} // namespace isae

#endif // LINE3DLANDMARKINITIALIZER_H
