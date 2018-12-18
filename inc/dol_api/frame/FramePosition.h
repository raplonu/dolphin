#ifndef DOLPHIN_FRAME_FRAME_POSITION_H
#define DOLPHIN_FRAME_FRAME_POSITION_H

#include <ci>

namespace dol {

  // Position handler
  struct RoiPos {
    const ci::Pos16 * roiPos;
    int validNb;

    CI_HODE RoiPos(const ci::Pos16 *roiPos, int validNb)
        : roiPos(roiPos), validNb(validNb) {}

    CI_HODE ci::Pos16 pos(int wfsId, int roiId) const {
      return roiPos[wfsId * validNb + roiId];
    }
  };

  struct PixPosPyr {
    const ci::Pos16 * pixPos;

    CI_HODE PixPosPyr(const ci::Pos16 *pixPos) : pixPos(pixPos) {}

    CI_HODE ci::Pos16 pos(int pixId) const { return pixPos[pixId]; }
  };

  struct PixPosSH {
    int subApSize;

    CI_HODE PixPosSH(int subApSize) : subApSize(subApSize) {}

    CI_HODE ci::Pos16 pos(int pixId) const {
      return ci::Pos16(pixId % subApSize, pixId / subApSize);
    }
  };

  template <typename PixPos>
  struct WFSPos {
    RoiPos roiPos;
    PixPos pixPos;

    CI_HODE WFSPos(ci::Pos16 const *roiPos, int validNb, PixPos pixPos)
        : roiPos(roiPos, validNb), pixPos(pixPos) {}

    CI_HODE ci::Pos16 pos(int wfsId, int roiId, int pixId) const {
      return roiPos.pos(wfsId, roiId) + pixPos.pos(pixId);
    }
  };

// Frame handler
template <typename PixPos>
struct FramePosition {
    WFSPos<PixPos> wfsPos;

    int wfsNb_;
    int frameSize_;

    CI_HODE FramePosition(WFSPos<PixPos> wfsPos, int wfsNb, int frameSize) :
        wfsPos(wfsPos), wfsNb_(wfsNb), frameSize_(frameSize)
    {}

    CI_HODE ci::Pos16 wfsOffset(int wfsId) const {
        return ci::Pos16(0, frameSize_ * wfsId);
    }

    CI_HODE ci::Pos16 posAt(int wfsId, int roiId, int pixId) const {
        return wfsPos.pos(wfsId, roiId, pixId) + wfsOffset(wfsId);
    }

    CI_HODE int offsetAt(int wfsId, int roiId, int pixId) const {
        ci::Pos16 pos = posAt(wfsId, roiId, pixId);
        return pos.x + pos.y * frameSize_;
    }

    CI_HODE constexpr int wfsNb() const noexcept {
      return wfsNb_;
    }

    CI_HODE constexpr int frameSize() const noexcept {
      return frameSize_;
    }

    CI_HODE constexpr int validNb() const noexcept {
      return wfsPos.roiPos.validNb;
    }

    CI_HODE constexpr const PixPos & pixelHandler() const noexcept {
      return wfsPos.pixPos;
    }

  };

  using SHPosition = FramePosition<PixPosSH>;
  using PyrPosition = FramePosition<PixPosPyr>;

}  // namespace dol

#endif  // DOLPHIN_FRAME_FRAME_POSITION_H


// template <typename PixPos>
// struct Frame {
//   WFSPos<PixPos> wfsPos_;

//   int wfsNb_;
//   int frameSize_;

//   CI_HODE Frame(WFSPos<PixPos> wfsPos, int wfsNb, int frameSize) :
//     wfsPos_(wfsPos), wfsNb_(wfsNb), frameSize_(frameSize)
//   {}

//   CI_HODE ci::Pos16 wfsOffset(int wfsId) const {
//     return ci::Pos16(0, frameSize_ * wfsId);
//   }

//   CI_HODE ci::Pos16 posAt(int wfsId, int roiId, int pixId) const {
//     return wfsPos_.pos(wfsId, roiId, pixId) + wfsOffset(wfsId);
//   }

//   CI_HODE int offsetAt(int wfsId, int roiId, int pixId) const {
//     ci::Pos16 pos = posAt(wfsId, roiId, pixId);
//     return pos.x + pos.y * frameSize_;
//   }

//   template<typename Ptr>
//   CI_HODE Ptr pixel(Ptr frame, int wfsId, int roiId, int pixId) const {
//     return static_cast<DDataT>(frame + offsetAt(wfsId, roiId, pixId));
//   }

//   template<typename Loader>
//   CI_HODE DDataT pixel(Loader loader, int wfsId, int roiId, int pixId) const {
//     return static_cast<DDataT>(loader(frameBuffer_.ptr() + offsetAt(wfsId, roiId, pixId)));
//   }

//   // template<typename Reseter>
//   // CI_DEVICE void reset(Reseter reseter) const {
//   //   if(ID < frameSize_ * frameSize_ * wfsNb_)
//   //       reseter(frameBuffer_.ptr() + ID);
//   // }
// };