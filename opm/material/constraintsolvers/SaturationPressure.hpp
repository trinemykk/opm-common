// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  Copyright 2026 SINTEF Digital

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/*!
 * \file
 * \copydoc Opm::SaturationPressure
 */
#ifndef OPM_SATURATION_PRESSURE_HPP
#define OPM_SATURATION_PRESSURE_HPP

#include <opm/input/eclipse/EclipseState/Compositional/CompositionalConfig.hpp>

#include <opm/material/fluidstates/CompositionalFluidState.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace Opm {

/*!
 * \brief Computes the saturation pressure of a mixture at a given temperature
 *        from the cubic equation of state.
 *
 * The bubble-point (dew-point) pressure of a liquid (vapour) with composition
 * \c z is the pressure where the incipient vapour (liquid) phase appears.  The
 * equilibrium ratios K are obtained by successive substitution on the fugacity
 * coefficient ratios at fixed pressure, and the pressure is updated to drive
 * the total amount of the incipient phase, sum_c K_c z_c (or sum_c z_c / K_c),
 * towards one.  The outer iteration scans a pressure branch from its nominal
 * single-phase side.  A converged trivial solution K == 1 advances the scan and
 * restarts the substitution from the Wilson estimate; other trials update the
 * pressure from the amount of the incipient phase.
 *
 * A mixture may have lower and upper dew points at a given temperature.  The
 * upper (retrograde) one is the boundary crossed when reservoir pressure
 * declines, and it is the one the reference simulator reports as the
 * saturation pressure of a gas.  The dew-point search therefore tries the
 * upper branch first and falls back to the lower branch if that search reports
 * no root.
 */
template <class Scalar, class FluidSystem>
class SaturationPressure
{
    static constexpr int numComponents = FluidSystem::numComponents;
    static constexpr int oilPhaseIdx = FluidSystem::oilPhaseIdx;
    static constexpr int gasPhaseIdx = FluidSystem::gasPhaseIdx;

    using EOSType = CompositionalConfig::EOSType;

public:
    using CompVec = std::array<Scalar, numComponents>;

    /// Computes the bubble-point pressure of a liquid with composition \p liquid at
    /// temperature \p temp, along with the equilibrium \p vapor composition.
    /// On failure \p press is left untouched while \p vapor holds iteration
    /// scratch and must not be read.
    /// \return whether the calculation converged
    static bool bubblePressure(const CompVec& liquid,
                               const Scalar temp,
                               const EOSType eosType,
                               Scalar& press,
                               CompVec& vapor)
    { return solve_(liquid, temp, eosType, Mode::Bubble, press, vapor) == Outcome::Converged; }

    /// Computes the dew-point pressure of a vapour with composition \p vapor at
    /// temperature \p temp, along with the equilibrium \p liquid composition.
    /// The upper (retrograde) dew point is preferred; the lower one is only
    /// searched when the upper branch reports no root, so a plain
    /// convergence failure reports false rather than the wrong branch.
    /// On failure \p press is left untouched while \p liquid holds iteration
    /// scratch and must not be read.
    /// \return whether the calculation converged
    static bool dewPressure(const CompVec& vapor,
                            const Scalar temp,
                            const EOSType eosType,
                            Scalar& press,
                            CompVec& liquid)
    {
        // Try the lower branch only when the upper-branch search reports that
        // it found no root. After a plain convergence failure it stays untried
        // because the mixture may still have both dew points.
        switch (solve_(vapor, temp, eosType, Mode::DewUpper, press, liquid)) {
        case Outcome::Converged:
            return true;
        case Outcome::NoRoot:
            return solve_(vapor, temp, eosType, Mode::DewLower, press, liquid)
                == Outcome::Converged;
        case Outcome::GaveUp:
            return false;
        }
        return false;
    }

private:
    // What a branch search returned: a converged saturation point, no candidate
    // root after the scan, or an exhausted iteration.
    enum class Outcome { Converged, NoRoot, GaveUp };

    // The saturation-pressure branch being searched.  The bubble point and the
    // upper (retrograde) dew point are approached from the high-pressure side,
    // the lower dew point from the low-pressure side.
    enum class Mode { Bubble, DewUpper, DewLower };

    // The Wilson correlation for K_c * press.
    static CompVec wilsonKp_(const Scalar temp)
    {
        CompVec Kp;
        for (int c = 0; c < numComponents; ++c) {
            Kp[c] = FluidSystem::criticalPressure(c) *
                    std::exp(5.373 * (1.0 + FluidSystem::acentricFactor(c)) *
                             (1.0 - FluidSystem::criticalTemperature(c) / temp));
        }
        return Kp;
    }

    static Outcome solve_(const CompVec& z,
                          const Scalar temp,
                          const EOSType eosType,
                          const Mode mode,
                          Scalar& press,
                          CompVec& incipient)
    {
        const CompVec wilsonKp = wilsonKp_(temp);
        const bool bubble = (mode == Mode::Bubble);
        // The bubble point and the retrograde dew point are approached from
        // the high-pressure side, the lower dew point from the low-pressure
        // side.  In every case the single-phase side is the one the scan
        // starts on, and the pressure update moves away from it.
        const bool fromAbove = (mode != Mode::DewLower);

        // Wilson K scales as 1/p, so its saturation sum supplies the initial
        // pressure. The upper dew search starts from the same high estimate as
        // the bubble search.
        Scalar p = 0.0;
        if (mode == Mode::DewLower) {
            for (int c = 0; c < numComponents; ++c) {
                p += z[c] / wilsonKp[c];
            }
            p = 1.0 / p;
        }
        else {
            for (int c = 0; c < numComponents; ++c) {
                p += z[c] * wilsonKp[c];
            }
        }

        auto wilsonK = [&wilsonKp](const Scalar pressure) {
            CompVec K;
            std::ranges::transform(wilsonKp, K.begin(),
                                   [pressure](const Scalar Kp) { return Kp / pressure; });
            return K;
        };
        CompVec K = wilsonK(p);

        // The known phase holds z; the incipient phase composition is derived from K.
        const auto knownPhaseIdx = bubble ? oilPhaseIdx : gasPhaseIdx;
        const auto incipientPhaseIdx = bubble ? gasPhaseIdx : oilPhaseIdx;

        CompositionalFluidState<Scalar, FluidSystem> fs;
        fs.setTemperature(temp);
        for (int c = 0; c < numComponents; ++c) {
            fs.setMoleFraction(knownPhaseIdx, c, z[c]);
        }

        // pSingle records a converged trivial-root trial; pTwo records a
        // converged trial with distinct EOS volumes. Once both are present,
        // use their geometric mean to limit later pressure updates.
        Scalar pSingle{}, pTwo{};
        bool haveSingle = false;
        bool haveTwo = false;

        // Use a 10% step while scanning a branch. A smaller step reduces the
        // chance of crossing a narrow phase envelope in one update.
        constexpr Scalar scanStep = 0.9;
        constexpr int maxOuter = 200;
        constexpr int maxInner = 100;

        for (int outer = 0; outer < maxOuter; ++outer) {
            fs.setPressure(oilPhaseIdx, p);
            fs.setPressure(gasPhaseIdx, p);

            // Fugacity equality at fixed pressure: K_c = phi_liquid / phi_vapour.
            bool trivial = false;
            bool rootsDistinct = false;
            bool substitutionConverged = false;
            for (int inner = 0; inner < maxInner; ++inner) {
                Scalar sum = 0.0;
                for (int c = 0; c < numComponents; ++c) {
                    incipient[c] = bubble ? K[c] * z[c] : z[c] / K[c];
                    sum += incipient[c];
                }
                for (int c = 0; c < numComponents; ++c) {
                    fs.setMoleFraction(incipientPhaseIdx, c, incipient[c] / sum);
                }

                typename FluidSystem::template ParameterCache<Scalar> paramCache(eosType);
                paramCache.updatePhase(fs, oilPhaseIdx);
                paramCache.updatePhase(fs, gasPhaseIdx);

                // A pure component or an azeotrope has K = 1 at a genuine
                // saturation point, where the two phases share a composition
                // but occupy different EOS roots.  The molar volumes tell that
                // state apart from the trivial solution, whose phases are one
                // and the same.
                const Scalar vmL = paramCache.molarVolume(oilPhaseIdx);
                const Scalar vmV = paramCache.molarVolume(gasPhaseIdx);
                // A molar volume near the cubic EOS floor does not establish a
                // physical second root.
                constexpr Scalar clampedVm = 1.0e-7;
                rootsDistinct = (std::min(vmL, vmV) > 2.0 * clampedVm) &&
                                (std::abs(vmL - vmV) > 1.0e-9 * std::max(vmL, vmV));

                Scalar change = 0.0;
                trivial = true;
                for (int c = 0; c < numComponents; ++c) {
                    const Scalar phiL = FluidSystem::fugacityCoefficient(
                        fs, paramCache, oilPhaseIdx, c);
                    const Scalar phiV = FluidSystem::fugacityCoefficient(
                        fs, paramCache, gasPhaseIdx, c);
                    const Scalar newK = phiL / phiV;
                    // Relative to the magnitude of K: an absolute measure is
                    // unreachable for the large K of a light component.
                    change = std::max(change, std::abs(newK - K[c]) /
                                              std::max(Scalar{1}, std::abs(newK)));
                    trivial = trivial && (std::abs(newK - 1.0) < 1.0e-5);
                    K[c] = newK;
                }
                if (change < 1.0e-12) {
                    substitutionConverged = true;
                    break;
                }
            }

            // The trivial test needs a converged K just as the pressure
            // criterion below does: its threshold is 1e-5 while convergence
            // is 1e-12, so an exhausted iteration can read as near-trivial
            // while K is still moving, and would then record an endpoint from
            // a pressure it never classified.
            if (substitutionConverged && trivial && !rootsDistinct) {
                // The phases collapsed onto one root. Record the trial as the
                // scan's nominal single-phase endpoint. If a distinct-root
                // trial is also available, use their geometric mean;
                // otherwise continue the branch scan.
                pSingle = p;
                haveSingle = true;
                p = haveTwo ? std::sqrt(pSingle * pTwo)
                            : p * (fromAbove ? scanStep : Scalar{1} / scanStep);
                K = wilsonK(p);
                continue;
            }

            Scalar sum = 0.0;
            for (int c = 0; c < numComponents; ++c) {
                sum += bubble ? K[c] * z[c] : z[c] / K[c];
            }
            // The pressure criterion is only meaningful once the fixed-pressure
            // substitution has actually reached fugacity equality; exhausting
            // the inner loop is a failure, not a solution.
            if (substitutionConverged && std::abs(sum - 1.0) < 1.0e-10) {
                Scalar distance = 0.0;
                for (int c = 0; c < numComponents; ++c) {
                    incipient[c] = (bubble ? K[c] * z[c] : z[c] / K[c]) / sum;
                    distance += std::abs(incipient[c] - z[c]);
                }
                // A converged K that leaves the incipient phase with both the
                // composition and the EOS root of the known one is the trivial
                // solution wearing a disguise: it satisfies the saturation
                // condition at an arbitrary pressure.  Treat it as the
                // non-solution and keep searching. Mixtures where the roots
                // have merged are refused for the same reason.
                if (distance > 1.0e-3 || rootsDistinct) {
                    press = p;
                    return Outcome::Converged;
                }
                pSingle = p;
                haveSingle = true;
                p = haveTwo ? std::sqrt(pSingle * pTwo)
                            : p * (fromAbove ? scanStep : Scalar{1} / scanStep);
                K = wilsonK(p);
                continue;
            }

            // Retain converged substitutions with distinct EOS volumes as the
            // candidate endpoint used to limit later updates. Distinct roots
            // alone do not test thermodynamic stability.
            if (substitutionConverged && rootsDistinct) {
                pTwo = p;
                haveTwo = true;
            }
            // Advance towards unit incipient amount: up on the bubble and
            // upper-dew branches, down on the lower-dew branch.
            const Scalar factor = (mode == Mode::DewLower) ? 1.0 / sum : sum;
            Scalar pNext = p * std::clamp(factor, Scalar{0.5}, Scalar{2.0});
            // Do not step past the recorded trivial-root endpoint; use the
            // geometric mean instead.
            if (haveSingle && ((fromAbove && pNext >= pSingle) ||
                               (!fromAbove && pNext <= pSingle)))
            {
                pNext = std::sqrt(p * pSingle);
            }
            p = pNext;
        }

        // If no distinct-root endpoint was recorded, report NoRoot only when
        // the scan encountered a converged trivial-root state. If every
        // substitution exhausted, the branch search merely gave up.
        if (!haveTwo) {
            return haveSingle ? Outcome::NoRoot : Outcome::GaveUp;
        }
        // Treat coincident recorded endpoints without an accepted solution as
        // no root. Distinct endpoints mean the search is unfinished.
        if (haveSingle &&
            (std::abs(pSingle - pTwo) <= 1.0e-6 * std::max(pSingle, pTwo)))
        {
            return Outcome::NoRoot;
        }
        return Outcome::GaveUp;
    }
};

} // namespace Opm

#endif // OPM_SATURATION_PRESSURE_HPP
