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
 * coefficient ratios at fixed pressure, accelerated by dominant-eigenvalue
 * extrapolation where the substitution contracts slowly, and the pressure is
 * updated to drive the total amount of the incipient phase, sum_c K_c z_c (or
 * sum_c z_c / K_c), towards one.  The outer iteration scans a pressure branch
 * from its nominal single-phase side.  A converged trivial solution K == 1
 * advances the scan and restarts the substitution from the Wilson estimate.
 * A converged non-trivial trial is classified by its incipient amount -- above
 * one the pressure lies inside the two-phase region, below one outside it --
 * and the search steps by a secant on the logarithms until those two sides
 * bracket the root, then by regula falsi within the bracket.  A candidate root
 * is accepted only after a tangent-plane stability test of the known phase in
 * the direction the search does not probe, which refuses stationary points
 * where the incipient phase merely occupies the other EOS root of an almost
 * identical composition inside the two-phase region.
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

    /// Whether the known phase \p z is unstable at the pressure and
    /// temperature held in \p fs, judged by a tangent-plane trial in the
    /// direction the saturation search does not probe: a phase of the same
    /// kind as \p z, started from the Wilson estimate, evaluated at the same
    /// EOS root.  A trial that converges to a stationary point with a total
    /// amount above one lies below the tangent plane, so \p z is not at a
    /// saturation point but inside the two-phase region.  A trial that does
    /// not converge cannot certify the phase and reports it unstable, so
    /// that no uncertified root is returned.
    static bool knownPhaseUnstable_(const CompositionalFluidState<Scalar, FluidSystem>& fs,
                                    const CompVec& z,
                                    const unsigned knownPhaseIdx,
                                    const CompVec& wilsonK,
                                    const bool knownIsLiquid,
                                    const EOSType eosType)
    {
        // The known phase keeps its own state; the trial takes its root type
        // through the same phase index in a second state.
        typename FluidSystem::template ParameterCache<Scalar> knownCache(eosType);
        knownCache.updatePhase(fs, knownPhaseIdx);
        CompVec lnPhiZ;
        for (int c = 0; c < numComponents; ++c) {
            lnPhiZ[c] = std::log(FluidSystem::fugacityCoefficient(fs, knownCache, knownPhaseIdx, c));
        }

        // A liquid is tested against a heavier liquid, a vapour against a
        // lighter vapour: the direction the incipient-phase search leaves out.
        CompVec Y;
        for (int c = 0; c < numComponents; ++c) {
            Y[c] = knownIsLiquid ? z[c] / wilsonK[c] : z[c] * wilsonK[c];
        }

        auto trial = fs;
        // Dominant-eigenvalue extrapolation, as in the saturation search.
        CompVec dPrev{};
        bool haveDPrev = false;
        constexpr int maxIter = 500;
        for (int iter = 0; iter < maxIter; ++iter) {
            Scalar sumY = 0.0;
            for (int c = 0; c < numComponents; ++c) {
                sumY += Y[c];
            }
            for (int c = 0; c < numComponents; ++c) {
                trial.setMoleFraction(knownPhaseIdx, c, Y[c] / sumY);
            }
            typename FluidSystem::template ParameterCache<Scalar> trialCache(eosType);
            trialCache.updatePhase(trial, knownPhaseIdx);

            Scalar change = 0.0;
            Scalar sumNew = 0.0;
            CompVec d{};
            for (int c = 0; c < numComponents; ++c) {
                if (z[c] <= 0.0) {
                    Y[c] = 0.0;
                    continue;
                }
                const Scalar lnPhiY =
                    std::log(FluidSystem::fugacityCoefficient(trial, trialCache, knownPhaseIdx, c));
                const Scalar Ynew = std::exp(std::log(z[c]) + lnPhiZ[c] - lnPhiY);
                change = std::max(change, std::abs(Ynew - Y[c]) / std::max(Scalar{1}, Ynew));
                d[c] = std::log(Ynew) - std::log(Y[c]);
                Y[c] = Ynew;
                sumNew += Ynew;
            }
            if (change < 1.0e-10) {
                // At a saturation point the trial collapses back onto the
                // known phase and the total is one to rounding; inside the
                // envelope the excess is orders of magnitude larger.
                return sumNew > 1.0 + 1.0e-8;
            }
            if (haveDPrev && (iter % 4 == 3)) {
                Scalar num = 0.0;
                Scalar den = 0.0;
                for (int c = 0; c < numComponents; ++c) {
                    num += d[c] * dPrev[c];
                    den += dPrev[c] * dPrev[c];
                }
                if (den > 0.0) {
                    const Scalar ratio = std::min(num / den, Scalar{0.98});
                    if (ratio > 0.0) {
                        const Scalar remaining = ratio / (1.0 - ratio);
                        for (int c = 0; c < numComponents; ++c) {
                            if (z[c] > 0.0) {
                                Y[c] *= std::exp(remaining * d[c]);
                            }
                        }
                    }
                }
            }
            dPrev = d;
            haveDPrev = true;
        }
        return true;
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

        // Bracket of the saturation pressure in the scan direction.  pOut is
        // a pressure known to lie on the starting (single-phase) side of the
        // root: a trial that collapsed onto the trivial solution, or a
        // converged trial whose incipient amount fell short of one.  pIn is a
        // pressure known to lie inside the two-phase region: a converged trial
        // whose incipient amount exceeded one, which is the tangent-plane
        // instability condition at that stationary point.  Distinct EOS roots
        // alone do not tell which side of the boundary a trial is on.
        Scalar pOut{}, pIn{};
        bool haveOut = false;
        bool haveIn = false;
        // ln(sum) at the bracket ends.  A trivial trial fixes the side but
        // supplies no value, so fOut is only known after a converged one.
        Scalar fOut{}, fIn{};
        bool haveFOut = false;
        bool haveFIn = false;
        // The previous converged non-trivial trial, for the secant step, and
        // which bracket end the previous step replaced (Illinois damping).
        Scalar lnpPrev{}, fPrev{};
        bool havePrev = false;
        int lastSide = 0;
        // Whether any converged trial other than the trivial solution was met.
        bool sawNonTrivial = false;

        // Use a 10% step while scanning a branch. A smaller step reduces the
        // chance of crossing a narrow phase envelope in one update.
        constexpr Scalar scanStep = 0.9;
        constexpr int maxOuter = 200;
        constexpr int maxInner = 500;

        for (int outer = 0; outer < maxOuter; ++outer) {
            fs.setPressure(oilPhaseIdx, p);
            fs.setPressure(gasPhaseIdx, p);

            // Fugacity equality at fixed pressure: K_c = phi_liquid / phi_vapour.
            bool trivial = false;
            bool rootsDistinct = false;
            bool substitutionConverged = false;
            // The previous change of ln K, for the dominant-eigenvalue
            // extrapolation below.
            CompVec dPrev{};
            bool haveDPrev = false;
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
                CompVec d;
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
                    d[c] = std::log(newK) - std::log(K[c]);
                    K[c] = newK;
                }
                if (change < 1.0e-12) {
                    substitutionConverged = true;
                    break;
                }

                // Successive substitution contracts ln K geometrically, with a
                // ratio that approaches one near a critical point, where it
                // needs hundreds of passes.  Every fourth pass, estimate that
                // ratio from the last two changes and extrapolate the
                // remaining geometric series in one jump.  Extrapolation is
                // withheld when the changes are not aligned, and capped so
                // that a ratio near one cannot throw K across the envelope.
                if (haveDPrev && (inner % 4 == 3)) {
                    Scalar num = 0.0;
                    Scalar den = 0.0;
                    for (int c = 0; c < numComponents; ++c) {
                        num += d[c] * dPrev[c];
                        den += dPrev[c] * dPrev[c];
                    }
                    if (den > 0.0) {
                        constexpr Scalar maxRatio = 0.98;
                        const Scalar ratio = std::min(num / den, maxRatio);
                        if (ratio > 0.0) {
                            const Scalar remaining = ratio / (1.0 - ratio);
                            for (int c = 0; c < numComponents; ++c) {
                                K[c] *= std::exp(remaining * d[c]);
                            }
                        }
                    }
                }
                dPrev = d;
                haveDPrev = true;
            }

            // The trivial test needs a converged K just as the pressure
            // criterion below does: its threshold is 1e-5 while convergence
            // is 1e-12, so an exhausted iteration can read as near-trivial
            // while K is still moving, and would then record an endpoint from
            // a pressure it never classified.
            const auto markOutside = [&](const bool withValue, const Scalar value) {
                pOut = p;
                haveOut = true;
                haveFOut = withValue;
                fOut = value;
            };
            // Step from a trial the substitution could not classify: bisect
            // the bracket if there is one, otherwise continue the scan.
            const auto stepUnclassified = [&]() {
                p = haveIn ? std::sqrt(pOut * pIn)
                           : p * (fromAbove ? scanStep : Scalar{1} / scanStep);
                K = wilsonK(p);
            };

            if (substitutionConverged && trivial && !rootsDistinct) {
                // The phases collapsed onto one root: the pressure lies on
                // the single-phase side of the boundary.
                markOutside(false, Scalar{0});
                lastSide = 0;
                stepUnclassified();
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
                //
                // Distinct roots also admit a stationary point where the
                // incipient phase merely occupies the other EOS root of an
                // almost identical composition inside the two-phase region.
                // Certify a candidate by the stability of the known phase:
                // at a saturation point it is marginally stable, inside the
                // envelope a trial phase of its own kind lies below its
                // tangent plane.
                if ((distance > 1.0e-3 || rootsDistinct) &&
                    !knownPhaseUnstable_(fs, z, knownPhaseIdx, wilsonK(p), bubble, eosType))
                {
                    press = p;
                    return Outcome::Converged;
                }
                // A refuted candidate is a stationary point that is not a
                // saturation point, like the disguised trivial solution.
                // It is recorded as a scan endpoint rather than a bracket
                // end: the bracket then collapses onto it and the branch
                // reports no root.
                markOutside(false, Scalar{0});
                lastSide = 0;
                stepUnclassified();
                continue;
            }

            // The incipient amount is the residual of the search: ln(sum) is
            // positive inside the two-phase region and negative outside it,
            // and the root is where it vanishes.
            const Scalar f = std::log(sum);
            const Scalar lnp = std::log(p);

            if (!substitutionConverged) {
                // An exhausted substitution classifies nothing.  Take the
                // fixed-point step K ~ 1/p implies, p * sum towards the root,
                // and let the carried-over K keep converging.
                const Scalar lnpFixed = lnp + (fromAbove ? f : -f);
                p = std::exp(std::clamp(lnpFixed - lnp, std::log(Scalar{0.5}), std::log(Scalar{2})) + lnp);
                if (haveOut && ((fromAbove && p >= pOut) || (!fromAbove && p <= pOut))) {
                    p = std::sqrt(std::exp(lnp) * pOut);
                }
                continue;
            }

            sawNonTrivial = true;
            if (f > 0) {
                // Inside the two-phase region.  Illinois: replacing the same
                // end twice in a row halves the value kept at the other end,
                // so regula falsi cannot stall against it.
                if (haveIn && lastSide > 0 && haveFOut) {
                    fOut *= 0.5;
                }
                pIn = p;
                fIn = f;
                haveIn = true;
                haveFIn = true;
                lastSide = 1;
            }
            else {
                if (haveOut && haveFOut && lastSide < 0 && haveIn && haveFIn) {
                    fIn *= 0.5;
                }
                markOutside(true, f);
                lastSide = -1;
            }

            // Propose the next pressure in ln p.  Regula falsi within a bracket
            // whose ends both carry a value; a secant through the previous
            // trial before the bracket closes, provided the slope has the sign
            // the branch implies (sum grows towards the root: falling pressure
            // from above, rising pressure from below); and the fixed-point
            // step otherwise.
            Scalar lnpNext{};
            bool proposed = false;
            if (haveIn && haveOut && haveFIn && haveFOut) {
                const Scalar lnpIn = std::log(pIn);
                const Scalar lnpOut = std::log(pOut);
                if (fIn != fOut) {
                    lnpNext = lnpOut - fOut * (lnpIn - lnpOut) / (fIn - fOut);
                    proposed = true;
                }
            }
            if (!proposed && havePrev && lnp != lnpPrev) {
                const Scalar slope = (f - fPrev) / (lnp - lnpPrev);
                const bool slopeOk = fromAbove ? (slope < -1.0e-12) : (slope > 1.0e-12);
                if (slopeOk) {
                    lnpNext = lnp - f / slope;
                    proposed = true;
                }
            }
            if (!proposed) {
                lnpNext = lnp + (fromAbove ? f : -f);
            }
            lnpPrev = lnp;
            fPrev = f;
            havePrev = true;

            // Keep every step within a factor of two of the current pressure,
            // so that a mixture with both dew points is not carried across the
            // whole envelope in one update, and keep it inside the bracket.
            lnpNext = lnp + std::clamp(lnpNext - lnp, std::log(Scalar{0.5}), std::log(Scalar{2}));
            Scalar pNext = std::exp(lnpNext);
            if (haveIn && haveOut) {
                const Scalar lo = std::min(pIn, pOut);
                const Scalar hi = std::max(pIn, pOut);
                if (!(pNext > lo && pNext < hi)) {
                    pNext = std::sqrt(lo * hi);
                }
            }
            else if (haveOut && ((fromAbove && pNext >= pOut) ||
                                 (!fromAbove && pNext <= pOut)))
            {
                pNext = std::sqrt(p * pOut);
            }
            p = pNext;
        }

        // Nothing inside the two-phase region was met.  That says the branch
        // has no root only if the scan classified trials as trivial and never
        // converged to any other stationary point; a search that was homing
        // in on a root when the iterations ran out has merely given up, and
        // reporting no root would hand an upper-dew search to the lower
        // branch on no evidence.
        if (!haveIn) {
            return (haveOut && !sawNonTrivial) ? Outcome::NoRoot : Outcome::GaveUp;
        }
        // A bracket that collapsed onto a point where only the trivial
        // solution lives has pinned the phase boundary down without finding
        // a saturation point there; that too is evidence of no root.  A
        // bracket closed by a converged trial on the outside contains a sign
        // change and so a root the search failed to reach.
        if (haveOut && !haveFOut &&
            (std::abs(pOut - pIn) <= 1.0e-6 * std::max(pOut, pIn)))
        {
            return Outcome::NoRoot;
        }
        return Outcome::GaveUp;
    }
};

} // namespace Opm

#endif // OPM_SATURATION_PRESSURE_HPP
