#ifndef DOLPHIN_SH_CENTROID_H
#define DOLPHIN_SH_CENTROID_H

#include <dol_api/frame/FramePosition.h>

namespace dol {
    template<typename T>
    struct SHCentroid {
        /**
         * @brief Declare sub type to store slope X, Y and intensity
         * All member are always initialized to zero by design
         * 
         */
        struct SubApRes {
            T xSlope{}, ySlope{}, intensity{};
        };

        SHPosition frameIndex;
        const T * intensityRef;

        ci::WarpGroup warp;

        int roiPerBlockNb;
        int roiPerBlockLocalNb;
        int roiPerWarp;
        int roiPerWarpLocal;
        int roiStartId;

        int pixPerSubAp;

        T modulationFactor;


        CI_DEVICE SHCentroid(SHPosition frameIndex, const T * intensityRef, T modulationFactor) :
            frameIndex(frameIndex), intensityRef(intensityRef), warp(),
            roiPerBlockNb(itemPerBlockNb(frameIndex.validNb())),
            roiPerBlockLocalNb(itemPerBlockLocalNb(frameIndex.validNb())),
            roiPerWarp(itemPerGroupNb(roiPerBlockLocalNb, WARP_NB)),
            roiPerWarpLocal(itemPerGroupLocalNb(roiPerBlockLocalNb, warp.id, WARP_NB)),
            roiStartId(warp.id * roiPerWarp + BID * roiPerBlockNb),
            pixPerSubAp(frameIndex.pixelHandler().subApSize * frameIndex.pixelHandler().subApSize),
            modulationFactor(modulationFactor)
        {}

        CI_DEVICE void exportPixel(T xSlope, T ySlope, T intensity, T * slopeOut, T * intensityOut) {
            slopeOut[0] = xSlope;
            slopeOut[1] = ySlope;

            intensityOut[0] = intensity;
        }

        CI_DEVICE constexpr int globalIndex(int wfsId, int roiLocalId) noexcept {
            return roiLocalId + roiStartId + (warp.id * roiPerWarp) + wfsId * roiPerBlockNb;
        }

        CI_DEVICE constexpr int localIndex(int wfsId, int roiLocalId) noexcept {
            return roiLocalId + (warp.id * roiPerWarp) + wfsId * roiPerBlockNb;
        }

        CI_DEVICE auto globalExporter(T * slopeOut, T * intensityOut) {

            return [this, slopeOut, intensityOut] CI_DEVICE (T xSlope, T ySlope, T intensity, int wfsId, int roiLocalId) {
                int offset = globalIndex(wfsId, roiLocalId);
                exportPixel(xSlope, ySlope, intensity, slopeOut + (2 * offset), intensityOut + offset);
            };
        }

        CI_DEVICE auto sharedExporter(T * slopeOut, T * intensityOut) {
            return [this, slopeOut, intensityOut] CI_DEVICE (T xSlope, T ySlope, T intensity, int wfsId, int roiLocalId) {
                int offset = localIndex(wfsId, roiLocalId);
                exportPixel(xSlope, ySlope, intensity, slopeOut + (2 * offset), intensityOut + offset);
            };
        }

        template<typename Ptr, typename Loader>
        CI_DEVICE SubApRes slopeAt(Ptr frame, Loader && loader, int wfsId, int roiLocalId) {
            SubApRes res;

            if (roiLocalId < roiPerWarpLocal)
                for (int pixId(warp.pos); pixId < pixPerSubAp; pixId += ci::warpsize) {
                    T pixel = loader(frame + frameIndex.offsetAt(wfsId, roiLocalId + roiStartId, pixId));

                    res.xSlope    += pixel * (pixId % frameIndex.pixelHandler().subApSize);
                    res.ySlope    += pixel * (pixId / frameIndex.pixelHandler().subApSize);
                    res.intensity += pixel;
                }

            // Reduce values

            res.xSlope    = dol::warpReduce(res.xSlope);
            res.ySlope    = dol::warpReduce(res.ySlope);
            res.intensity = dol::warpReduce(res.intensity);

            auto halfsubs = (frameIndex.pixelHandler().subApSize - 1.) / 2;
            if(res.intensity >= intensityRef[globalIndex(wfsId, roiLocalId)])
                return SubApRes{
                    modulationFactor * (res.xSlope / res.intensity - halfsubs),
                    modulationFactor * (res.ySlope / res.intensity - halfsubs),
                    res.intensity};
            else
                return SubApRes{T{}, T{}, res.intensity};
        }

        template<typename Ptr, typename Loader, typename Exporter>
        CI_DEVICE void computeAt(Ptr frame, Loader && loader, int wfsId, int roiLocalId, Exporter exporter) {
            SubApRes res = slopeAt(frame, loader, wfsId, roiLocalId);

            if (warp.pos == 0 && roiLocalId < roiPerWarpLocal)
                exporter(res.xSlope, res.ySlope, res.intensity, wfsId, roiLocalId);
        }

        template<typename Ptr, typename Loader, typename Exporter>
        CI_DEVICE void computeGeneric(Ptr frame, Loader && loader, Exporter exporter) {
            for (int wfsId{}; wfsId < frameIndex.wfsNb(); ++wfsId) 
                for (int roiLocalId{}; roiLocalId < roiPerWarp; ++roiLocalId) 
                    computeAt(frame, loader, wfsId, roiLocalId, exporter);
        }

        template<typename Ptr, typename Loader>
        CI_DEVICE void globalCompute(Ptr frame, Loader && loader, T * slope, T * intensity) {
            computeGeneric(frame, std::forward<Loader>(loader), globalExporter(slope, intensity));
        }

        template<typename Ptr, typename Loader>
        CI_DEVICE void sharedCompute(Ptr frame, Loader loader, T * slope, T * intensity) {
            computeGeneric(frame, std::forward<Loader>(loader), sharedExporter(slope, intensity));
        }

    };
}

#endif //DOLPHIN_SH_CENTROID_H