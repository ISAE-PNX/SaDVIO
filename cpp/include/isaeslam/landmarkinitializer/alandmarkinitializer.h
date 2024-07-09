#ifndef ALANDMARKINITIALIZER_H
#define ALANDMARKINITIALIZER_H

#include <iostream>
#include <opencv2/core.hpp>

#include "isaeslam/data/landmarks/ALandmark.h"
#include "isaeslam/typedefs.h"

namespace isae {

class ImageSensor;
class AFeature;
class ALandmark;

class ALandmarkInitializer : public std::enable_shared_from_this<ALandmarkInitializer> {
  public:
    ALandmarkInitializer(uint nb_requiered_ldmk)
        : _nb_requiered_ldmk(nb_requiered_ldmk) {}
    ~ALandmarkInitializer() {}

    typed_vec_landmarks getInitializedLandmarks() { return _initialized_landmarks; }

    uint initFromMatch(feature_pair match);
    uint initFromMatches(vec_match matches);
    uint initFromFeatures(std::vector<std::shared_ptr<AFeature>> feats);
    uint getNbRequieredLdmk() { return _nb_requiered_ldmk; }

  protected:
    std::shared_ptr<ALandmark> createNewLandmark(std::shared_ptr<AFeature> f1, std::shared_ptr<AFeature> f2);
    typed_vec_landmarks _initialized_landmarks;

  private:
    uint _nb_requiered_ldmk;
    virtual bool initLandmark(std::vector<std::shared_ptr<AFeature>> features,
                              std::shared_ptr<ALandmark> &landmark)          = 0;
    virtual bool initLandmarkWithDepth(std::vector<std::shared_ptr<AFeature>> features,
                                       std::shared_ptr<ALandmark> &landmark) = 0;

    // must link f or f1 & f2 to the landmark
    virtual std::shared_ptr<ALandmark> createNewLandmark(std::shared_ptr<AFeature> f) = 0;
};

} // namespace isae

#endif // ALANDMARKINITIALIZER_H
