#ifndef EDGELET3DLANDMARKINITIALIZER_H
#define EDGELET3DLANDMARKINITIALIZER_H

#include "isaeslam/data/landmarks/Edgelet3D.h"
#include "isaeslam/landmarkinitializer/alandmarkinitializer.h"
#include "isaeslam/typedefs.h"

namespace isae {

class Edgelet3DLandmarkInitializer : public ALandmarkInitializer {
  public:
    Edgelet3DLandmarkInitializer(uint nb_requiered_ldmk)
        : ALandmarkInitializer(nb_requiered_ldmk) {}

  private:
    bool initLandmark(std::vector<std::shared_ptr<AFeature>> features, std::shared_ptr<ALandmark> &landmark) override;
    bool initLandmarkWithDepth(std::vector<std::shared_ptr<AFeature>> features,
                               std::shared_ptr<ALandmark> &landmark) override;

    // must link f or f1 & f2 to the landmark
    std::shared_ptr<ALandmark> createNewLandmark(std::shared_ptr<AFeature> f) override;

    static Eigen::Matrix3d processOrientation(std::vector<Eigen::Vector3d> Ns);
};

} // namespace isae

#endif // EDGELET3DLANDMARKINITIALIZER_H
