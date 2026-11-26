/*
  Copyright 2026 NORCE Research AS.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
/*!
 * \file
 * \brief Cross-step term of RESV (reservoir-voidage) well controls.
 *
 * A producer under RESV control converts its surface rates to reservoir
 * voidage with coefficients evaluated at a field-average PVT state
 *     a = (p, T, Rs, Rv, Rsw, Rvw, salt),
 * hydrocarbon-pore-volume weighted averages, except Rs (oil phase volume
 * weighted) and Rv (gas phase volume weighted) since OPM/opm-simulators
 * 3914e3f38 (2026-10-30), which BlackoilWellModel recomputes
 * (rateConverter_->defineState) at every report-step start and at the end of
 * every accepted time step (timeStepSucceeded). The control equation of
 * substep k,
 *     R_ctrl = -sum_p q_p coeff_p(a) - target(a)   (WCONHIST: target = voidage
 *                                                   of the history rates)
 *     R_ctrl = -sum_p q_p coeff_p(a) - resv_rate   (WCONPROD),
 * therefore depends on the converged state of substep k-1 through a(x_{k-1}).
 * The simulator treats a as a constant, so this explicit cross-step dependence
 * is absent from the Jacobian the adjoint reuses. In the Lagrangian it adds
 *     (da/dx_{k-1})^T g_k,  g_k = sum_w lambda_ctrl,k,w dR_ctrl,k,w/da,
 * to the stationarity condition of x_{k-1} (a right-hand-side term of
 * substep k-1, like the storage cross term Bdiag), and g_k . da/dpvmult to the
 * pore-volume gradient (the averaging weights are pore volumes).
 *
 * dR_ctrl/da is a central difference: the target through OPM's own
 * state-independent SurfaceToReservoirVoidage::calcReservoirVoidageRates, the
 * coefficients through a copy of SurfaceToReservoirVoidage::calcCoeff (the
 * state-independent overload is not instantiated in libopmsimulators) on
 * OPM's PVT objects. da/dx and da/dpvmult are exact (AD intensive quantities).
 * Self-checks, logged per sweep: the coefficients and target recomputed at
 * the averages of x_{k-1} must equal what the well model's converter (which
 * holds exactly those averages while substep k is replayed) returned, and
 * the control residual must vanish at the converged state.
 *
 * A second RESV inexactness sits within the step: the coefficients use
 * Rs = min(qg/qo, Rs_avg) and Rv = min(qo/qg, Rv_avg) with the well's own
 * surface rates, which the simulator holds fixed while differentiating, so
 * whenever a clamp is inactive the control row of the well Jacobian D lacks
 * -sum_p q_p sum_m dcoeff_p/dq_m dq_m/dx_w. correctJacobian() adds it to the
 * replayed D before the adjoint uses it.
 *
 * v1 scope: producers under RESV control (history and prediction mode), in
 * serial runs. RESV injectors and group RESV/VREP/REIN controls are not
 * covered.
 */
#ifndef OPM_ADJOINT_RESV_TERM_HPP
#define OPM_ADJOINT_RESV_TERM_HPP

#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/input/eclipse/Schedule/Schedule.hpp>
#include <opm/input/eclipse/Schedule/Well/Well.hpp>

#include <opm/models/utils/propertysystem.hh>

#include <opm/simulators/wells/MultisegmentWell.hpp>
#include <opm/simulators/wells/RateConverter.hpp>
#include <opm/simulators/wells/StandardWell.hpp>

#include <dune/common/dynvector.hh>
#include <dune/common/fvector.hh>
#include <dune/grid/common/partitionset.hh>
#include <dune/grid/common/rangegenerators.hh>
#include <dune/istl/bvector.hh>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace Opm {

template<class TypeTag>
class AdjointResvTerm
{
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Evaluation = GetPropType<TypeTag, Properties::Evaluation>;
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;
    using ElementContext = GetPropType<TypeTag, Properties::ElementContext>;
    static constexpr int numEq = getPropValue<TypeTag, Properties::NumEq>();
    using Vector = Dune::BlockVector<Dune::FieldVector<Scalar, numEq>>;
    using WellAdjoints = std::map<std::string, Dune::DynamicVector<Scalar>>;
    using RateConverterType =
        RateConverter::SurfaceToReservoirVoidage<FluidSystem, std::vector<int>>;

public:
    //! Components of the average state, in the order of the derivative vector.
    enum Avg : int { P, T, Rs, Rv, Rsw, Rvw, Salt, NumAvg };
    using AvgVector = std::array<Scalar, NumAvg>;

    explicit AdjointResvTerm(const bool enabled)
        : enabled_(enabled)
        , switchedOffByUser_(!enabled)
        , converter_(std::vector<int>(1, 0)) // state-independent calls only
    {}

    bool enabled() const
    { return enabled_; }

    //! \brief Leave the RESV control equations as the simulator linearizes
    //!        them (approximate gradients for RESV-controlled producers).
    void disable()
    { enabled_ = false; }

    //! \brief True if some producer of \p schedule has a RESV control or limit.
    static bool deckHasResvProducer(const Schedule& schedule)
    {
        for (std::size_t step = 0; step < schedule.size(); ++step) {
            for (const auto& name : schedule.wellNames(step)) {
                const auto& well = schedule.getWell(name, step);
                if (!well.isProducer()) {
                    continue;
                }
                const auto& properties = well.getProductionProperties();
                if (properties.controlMode == Well::ProducerCMode::RESV ||
                    properties.hasProductionControl(Well::ProducerCMode::RESV)) {
                    return true;
                }
            }
        }
        return false;
    }

    //! \brief Control equations recorded by collect() and not yet applied.
    bool pending() const
    { return !entries_.empty(); }

    //! \brief Exact RESV control rows in the replayed substep's well
    //!        Jacobians. The coefficients depend on the well's own surface
    //!        rates through the Rs/Rv clamp (whenever the well's ratio is
    //!        below the average one), and the simulator holds them fixed while
    //!        differentiating, so its control row lacks
    //!            -sum_p q_p sum_m dcoeff_p/dq_m dq_m/dx_w.
    //!        That term is added to row numWellEq-1 of D (segment-0 block of a
    //!        MultisegmentWell); dcoeff/dq is a central difference through the
    //!        well's own rate converter, which holds a(x_{k-1}) while substep k
    //!        is replayed, so no averages are recomputed here. Call right after
    //!        the substep is replayed, before D is used; \p refreshWellSolver
    //!        re-inverts D for the Schur-folded path.
    void correctJacobian(const Simulator& simulator, const bool refreshWellSolver)
    {
        if (!enabled_) {
            return;
        }
        const auto& wellModel = simulator.problem().wellModel();
        const auto& wellState = wellModel.wellState();
        const auto& summaryState = simulator.vanguard().summaryState();
        for (const auto& wellPtr : wellModel.localNonshutWells()) {
            if (!isResvProducer_(*wellPtr, wellState, summaryState)) {
                continue;
            }
            const auto& ws = wellState.well(wellPtr->name());
            if (const auto* sw =
                    dynamic_cast<const StandardWell<TypeTag>*>(wellPtr.get())) {
                correctControlRow_(*sw, ws.surface_rates, refreshWellSolver);
            } else if (const auto* mw =
                           dynamic_cast<const MultisegmentWell<TypeTag>*>(wellPtr.get())) {
                correctControlRow_(*mw, ws.surface_rates, refreshWellSolver);
            }
        }
    }

    //! \brief Record the RESV control equations of substep \p k. Call after
    //!        the well adjoints of k are known, while the well container and
    //!        rate converter are those of the replayed substep k (the
    //!        converter then holds the averages of x_{k-1}).
    void collect(const Simulator& simulator, const WellAdjoints& lambdaWells,
                 const int k)
    {
        entries_.clear();
        if (!enabled_) {
            return;
        }
        const auto& wellModel = simulator.problem().wellModel();
        const auto& wellState = wellModel.wellState();
        const auto& summaryState = simulator.vanguard().summaryState();
        const int numPhases = FluidSystem::numActivePhases();

        for (const auto& wellPtr : wellModel.localNonshutWells()) {
            if (!isResvProducer_(*wellPtr, wellState, summaryState)) {
                continue;
            }
            const auto& ws = wellState.well(wellPtr->name());
            const auto lw = lambdaWells.find(wellPtr->name());
            if (lw == lambdaWells.end()) {
                continue;
            }
            Entry e;
            e.well = wellPtr->name();
            e.substep = k;
            e.pvtRegion = wellPtr->pvtRegionIdx();
            // Control row: last well equation of a StandardWell, segment-0
            // pressure row of a MultisegmentWell -- index numWellEq-1 of the
            // (flattened) well adjoint either way.
            std::size_t numWellEq = 0;
            if (const auto* sw =
                    dynamic_cast<const StandardWell<TypeTag>*>(wellPtr.get())) {
                numWellEq = sw->linSys().getD()[0][0].N();
                surfaceRates_(sw->primaryVariables(), e.rates);
            } else if (const auto* mw =
                           dynamic_cast<const MultisegmentWell<TypeTag>*>(wellPtr.get())) {
                numWellEq = mw->linSys().getD()[0][0].N();
                surfaceRates_(mw->primaryVariables(), e.rates);
            } else {
                continue;
            }
            e.lambdaCtrl = lw->second[numWellEq - 1];

            const auto controls = wellPtr->wellEcl().productionControls(summaryState);
            e.history = !controls.prediction_mode;
            e.resvRate = controls.resv_rate;
            e.wsRates.assign(ws.surface_rates.begin(), ws.surface_rates.end());
            e.hrates = historyRates_(controls);

            // What the simulator used: the well's converter holds a(x_{k-1}).
            const auto& conv = wellPtr->rateConverter();
            e.coeffRef.assign(numPhases, 0.0);
            conv.calcCoeff(/*fipreg*/ 0, e.pvtRegion, e.wsRates, e.coeffRef);
            e.targetRef = e.resvRate;
            if (e.history) {
                std::vector<Scalar> resv(numPhases, 0.0);
                conv.calcReservoirVoidageRates(/*fipreg*/ 0, e.pvtRegion, e.hrates, resv);
                e.targetRef = std::accumulate(resv.begin(), resv.end(), Scalar{0});
            }
            entries_.push_back(std::move(e));
        }
        if (!entries_.empty()) {
            ++substepsWithTerm_;
        }
    }

    //! \brief Apply the recorded control equations of substep k+1, whose
    //!        averages are a(x_k), with x_k the state at time index 0:
    //!        rhs -= (da/dx_k)^T g and gradientPv += g . da/dpvmult. Pass
    //!        rhs = nullptr for the initial state (it has no adjoint
    //!        equation; only the pore-volume term applies).
    void apply(const Simulator& simulator, Vector* rhs,
               std::vector<Scalar>& gradientPv)
    {
        if (entries_.empty()) {
            return;
        }

        // Pass 1: the averages, exactly as SurfaceToReservoirVoidage::
        // defineState + sumRates: hydrocarbon pore volume weights (pore
        // volume weights if the region holds no hydrocarbon), then Rs by oil
        // and Rv by gas phase volume over the hydrocarbon cells, where the
        // region holds that phase.
        AvgVector sumHc{}, sumPv{};
        Scalar wHc = 0.0, wPv = 0.0;
        // {sum Rs pv So, sum pv So, sum Rv pv Sg, sum pv Sg}
        std::array<Scalar, 4> phaseSums{};
        forEachCell_(simulator, [&](unsigned, const CellQuantities& c) {
            const Scalar hcpv = c.pv.value() * c.hc.value();
            if (hcpv > 0.0) {
                wHc += hcpv;
                for (int q = 0; q < NumAvg; ++q) {
                    sumHc[q] += c.f[q].value() * hcpv;
                }
                const Scalar oilPv = c.pv.value() * c.so.value();
                const Scalar gasPv = c.pv.value() * c.sg.value();
                phaseSums[0] += c.f[Rs].value() * oilPv;
                phaseSums[1] += oilPv;
                phaseSums[2] += c.f[Rv].value() * gasPv;
                phaseSums[3] += gasPv;
            }
            if (c.pv.value() > 0.0) {
                wPv += c.pv.value();
                for (int q = 0; q < NumAvg; ++q) {
                    sumPv[q] += c.f[q].value() * c.pv.value();
                }
            }
        });
        const auto& comm = simulator.vanguard().grid().comm();
        wHc = comm.sum(wHc);
        wPv = comm.sum(wPv);
        comm.sum(sumHc.data(), NumAvg);
        comm.sum(sumPv.data(), NumAvg);
        comm.sum(phaseSums.data(), phaseSums.size());
        const bool useHc = wHc > 0.0;
        const bool rsByOil = useHc && phaseSums[1] > 0.0;
        const bool rvByGas = useHc && phaseSums[3] > 0.0;
        AvgVector W{};
        AvgVector a{};
        for (int q = 0; q < NumAvg; ++q) {
            W[q] = useHc ? wHc : wPv;
            a[q] = (useHc ? sumHc[q] : sumPv[q]) / W[q];
        }
        if (rsByOil) {
            W[Rs] = phaseSums[1];
            a[Rs] = phaseSums[0] / W[Rs];
        }
        if (rvByGas) {
            W[Rv] = phaseSums[3];
            a[Rv] = phaseSums[2] / W[Rv];
        }

        // g = sum_e lambda_ctrl dR_ctrl/da, plus the self-checks.
        AvgVector g{};
        for (const auto& e : entries_) {
            const auto coeff = coefficients_(e, a);
            Scalar scale = 0.0;
            for (int p = 0; p < 3; ++p) {
                if (FluidSystem::phaseIsActive(p)) {
                    const int pos = activePhase_(p);
                    scale += std::abs(e.rates[p] * coeff[pos]);
                    maxCoeffMismatch_ = std::max(maxCoeffMismatch_,
                        std::abs(coeff[pos] - e.coeffRef[pos]) /
                        std::max(std::abs(e.coeffRef[pos]), Scalar{1e-30}));
                }
            }
            const Scalar target = target_(e, a);
            maxTargetMismatch_ = std::max(maxTargetMismatch_,
                std::abs(target - e.targetRef) /
                std::max(std::abs(e.targetRef), Scalar{1e-30}));
            const Scalar r0 = residual_(e, a);
            maxResidual_ = std::max(maxResidual_,
                                    std::abs(r0) / std::max(scale, Scalar{1e-30}));
            countClampInactive_(e, a);

            for (int q = 0; q < NumAvg; ++q) {
                if (!varies_(q, a)) {
                    continue;
                }
                const Scalar h = step_(q, a[q]);
                AvgVector ap = a;
                ap[q] += h;
                Scalar dR;
                if (nonNegative_(q) && a[q] - h < 0.0) {
                    dR = (residual_(e, ap) - r0) / h;
                } else {
                    AvgVector am = a;
                    am[q] -= h;
                    dR = (residual_(e, ap) - residual_(e, am)) / (2.0 * h);
                }
                g[q] += e.lambdaCtrl * dR;
            }
            ++entriesApplied_;
        }
        for (int q = 0; q < NumAvg; ++q) {
            maxAbsG_[q] = std::max(maxAbsG_[q], std::abs(g[q]));
        }

        // Pass 2: da_q/dx_c = (w_q,c df_q,c + (f_q,c - a_q) dw_q,c) / W_q over
        // the cells of the average, and da_q/dpvmult_c = (f_q,c - a_q) w_q,c / W_q
        // (every weight is proportional to the pore volume).
        forEachCell_(simulator, [&](unsigned cellIdx, const CellQuantities& c) {
            const Evaluation wCell = useHc ? Evaluation(c.pv * c.hc) : c.pv;
            if (!(wCell.value() > 0.0)) {
                return;
            }
            Scalar pvTerm = 0.0;
            for (int q = 0; q < NumAvg; ++q) {
                if (g[q] == 0.0) {
                    continue;
                }
                const Evaluation w = (q == Rs && rsByOil) ? Evaluation(c.pv * c.so)
                                   : (q == Rv && rvByGas) ? Evaluation(c.pv * c.sg)
                                                          : wCell;
                const Scalar dev = c.f[q].value() - a[q];
                pvTerm += g[q] * dev * w.value() / W[q];
                if (rhs) {
                    for (int eq = 0; eq < numEq; ++eq) {
                        (*rhs)[cellIdx][eq] -= g[q] *
                            (w.value() * c.f[q].derivative(eq) +
                             dev * w.derivative(eq)) / W[q];
                    }
                }
            }
            gradientPv[cellIdx] += pvTerm;
        });
        entries_.clear();
    }

    //! \brief One-line account of the sweep (call once at its end).
    void logSummary() const
    {
        if (switchedOffByUser_) {
            OpmLog::info("Adjoint RESV control derivatives: disabled "
                         "(--adjoint-resv-derivatives=false)");
            return;
        }
        if (entriesApplied_ == 0 && rowsCorrected_ == 0) {
            return;
        }
        OpmLog::info(fmt::format(
            "Adjoint RESV controls: cross-step term over {} control equations in "
            "{} substeps (checks: max rel. coefficient mismatch {:.1e}, target "
            "mismatch {:.1e}, control residual {:.1e}; max|g| p {:.3e} T {:.3e} "
            "Rs {:.3e} Rv {:.3e}); Rs clamp inactive in {}, Rv clamp inactive in "
            "{}; {} control rows given the rate-dependence of the coefficients "
            "(max |correction|/|row| {:.2e})",
            entriesApplied_, substepsWithTerm_, maxCoeffMismatch_,
            maxTargetMismatch_, maxResidual_, maxAbsG_[P], maxAbsG_[T],
            maxAbsG_[Rs], maxAbsG_[Rv], rsClampInactive_, rvClampInactive_,
            rowsCorrected_, maxRowCorrection_));
    }

private:
    struct Entry
    {
        std::string well;
        int substep{-1};
        int pvtRegion{0};
        bool history{true};
        Scalar lambdaCtrl{0.0};
        std::array<Scalar, 3> rates{};   //!< canonical-phase primary-variable rates
        std::vector<Scalar> wsRates;     //!< well-state surface rates (active order)
        std::vector<Scalar> hrates;      //!< history surface rates (active order)
        Scalar resvRate{0.0};            //!< prediction-mode target
        std::vector<Scalar> coeffRef;    //!< coefficients the simulator used
        Scalar targetRef{0.0};           //!< target the simulator used
    };

    struct CellQuantities
    {
        Evaluation pv;                      //!< pore volume
        Evaluation hc;                      //!< 1 - Sw
        Evaluation so;                      //!< oil saturation (0 without oil and gas)
        Evaluation sg;                      //!< gas saturation (0 without oil and gas)
        std::array<Evaluation, NumAvg> f;   //!< p, T, Rs, Rv, Rsw, Rvw, salt
    };

    static int activePhase_(const unsigned canonicalPhase)
    {
        return FluidSystem::phaseIsActive(canonicalPhase)
            ? FluidSystem::canonicalToActivePhaseIdx(canonicalPhase) : -1;
    }

    //! History surface rates (active phase order) of WCONHIST controls.
    static std::vector<Scalar>
    historyRates_(const Well::ProductionControls& controls)
    {
        std::vector<Scalar> hrates(FluidSystem::numActivePhases(), 0.0);
        if (FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)) {
            hrates[activePhase_(FluidSystem::waterPhaseIdx)] = controls.water_rate;
        }
        if (FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx)) {
            hrates[activePhase_(FluidSystem::oilPhaseIdx)] = controls.oil_rate;
        }
        if (FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx)) {
            hrates[activePhase_(FluidSystem::gasPhaseIdx)] = controls.gas_rate;
        }
        return hrates;
    }

    //! An open producer whose active control is RESV with a nonzero target
    //! (a zero target makes OPM assemble WQTotal = 0 instead).
    template<class WellType, class WellStateType>
    static bool isResvProducer_(const WellType& well, const WellStateType& wellState,
                                const SummaryState& summaryState)
    {
        if (!well.isProducer() || well.wellIsStopped()) {
            return false;
        }
        if (wellState.well(well.name()).production_cmode != Well::ProducerCMode::RESV) {
            return false;
        }
        const auto controls = well.wellEcl().productionControls(summaryState);
        if (controls.prediction_mode) {
            return controls.resv_rate != 0.0;
        }
        const auto hrates = historyRates_(controls);
        return std::any_of(hrates.begin(), hrates.end(),
                           [](const Scalar r) { return r != 0.0; });
    }

    //! See correctJacobian(); \p wsRates are the well-state surface rates the
    //! simulator passed to the converter (active phase order).
    template<class WellType>
    void correctControlRow_(const WellType& well, const std::vector<Scalar>& wsRates,
                            const bool refreshWellSolver)
    {
        const int np = FluidSystem::numActivePhases();
        const auto& conv = well.rateConverter();
        const int pvt = well.pvtRegionIdx();
        const auto& pv = well.primaryVariables();

        // Rates with their well derivatives (active phase order) and the
        // canonical phase of each active slot.
        using EvalW = std::remove_cvref_t<decltype(pv.getQs(0))>;
        std::vector<EvalW> q;
        q.reserve(np);
        for (int ap = 0; ap < np; ++ap) {
            q.push_back(pv.getQs(0) * 0.0);
        }
        const std::array<std::pair<unsigned, unsigned>, 3> phaseComp{{
            {FluidSystem::waterPhaseIdx, FluidSystem::waterCompIdx},
            {FluidSystem::oilPhaseIdx, FluidSystem::oilCompIdx},
            {FluidSystem::gasPhaseIdx, FluidSystem::gasCompIdx}}};
        for (const auto& [phase, comp] : phaseComp) {
            if (FluidSystem::phaseIsActive(phase)) {
                q[activePhase_(phase)] =
                    pv.getQs(FluidSystem::canonicalToActiveCompIdx(comp));
            }
        }

        // s_m = sum_p q_p dcoeff_p/dq_m (central differences on the rates the
        // converter clamps with; zero when the clamps hold).
        Scalar qmax = 0.0;
        for (const auto v : wsRates) {
            qmax = std::max(qmax, std::abs(v));
        }
        if (qmax == 0.0) {
            return;
        }
        std::vector<Scalar> s(np, 0.0);
        std::vector<Scalar> cp(np), cm(np);
        bool any = false;
        for (int m = 0; m < np; ++m) {
            const Scalar h = 1e-7 * std::max(std::abs(wsRates[m]), 1e-6 * qmax);
            auto qp = wsRates;
            auto qm = wsRates;
            qp[m] += h;
            qm[m] -= h;
            conv.calcCoeff(/*fipreg*/ 0, pvt, qp, cp);
            conv.calcCoeff(/*fipreg*/ 0, pvt, qm, cm);
            for (int p = 0; p < np; ++p) {
                s[m] += q[p].value() * (cp[p] - cm[p]) / (2.0 * h);
            }
            any = any || s[m] != 0.0;
        }
        if (!any) {
            return;
        }

        auto& eqns = const_cast<std::remove_cvref_t<decltype(well.linSys())>&>(well.linSys());
        auto& block = const_cast<std::remove_cvref_t<decltype(eqns.getD()[0][0])>&>(
            eqns.getD()[0][0]);
        const std::size_t numWellEq = block.N();
        const std::size_t ctrl = numWellEq - 1;
        Scalar rowMax = 0.0, deltaMax = 0.0;
        for (std::size_t j = 0; j < numWellEq; ++j) {
            Scalar delta = 0.0;
            for (int m = 0; m < np; ++m) {
                delta -= s[m] * q[m].derivative(numEq + j);
            }
            rowMax = std::max(rowMax, std::abs(block[ctrl][j]));
            deltaMax = std::max(deltaMax, std::abs(delta));
            block[ctrl][j] += delta;
        }
        if (refreshWellSolver) {
            if constexpr (requires { eqns.invert(); }) {
                eqns.invert();
            } else {
                eqns.createSolver();
            }
        }
        ++rowsCorrected_;
        maxRowCorrection_ = std::max(maxRowCorrection_,
                                     deltaMax / std::max(rowMax, Scalar{1e-30}));
    }

    template<class PrimaryVariables>
    static void surfaceRates_(const PrimaryVariables& pv, std::array<Scalar, 3>& q)
    {
        q = {0.0, 0.0, 0.0};
        if (FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)) {
            q[FluidSystem::waterPhaseIdx] = pv.getQs(
                FluidSystem::canonicalToActiveCompIdx(FluidSystem::waterCompIdx)).value();
        }
        if (FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx)) {
            q[FluidSystem::oilPhaseIdx] = pv.getQs(
                FluidSystem::canonicalToActiveCompIdx(FluidSystem::oilCompIdx)).value();
        }
        if (FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx)) {
            q[FluidSystem::gasPhaseIdx] = pv.getQs(
                FluidSystem::canonicalToActiveCompIdx(FluidSystem::gasCompIdx)).value();
        }
    }

    //! The dissolved/vaporised ratio clamp of RateConverter.cpp.
    static std::pair<Scalar, Scalar>
    clampRatio_(const int io, const int ig, const Scalar rs, const Scalar rv,
                const std::vector<Scalar>& q)
    {
        if (io < 0 || ig < 0) {
            return {rs, rv};
        }
        auto eps = std::copysign(1.0e-15, q[io]);
        const Scalar rsq = q[ig] / (q[io] + eps);
        eps = std::copysign(1.0e-15, q[ig]);
        const Scalar rvq = q[io] / (q[ig] + eps);
        return {std::clamp(rsq, Scalar{0.0}, rs), std::clamp(rvq, Scalar{0.0}, rv)};
    }

    //! Copy of SurfaceToReservoirVoidage::calcCoeff(r, pvt, surface_rates,
    //! coeff) at averages \p a.
    std::vector<Scalar> coefficients_(const Entry& e, const AvgVector& a) const
    {
        const int iw = activePhase_(FluidSystem::waterPhaseIdx);
        const int io = activePhase_(FluidSystem::oilPhaseIdx);
        const int ig = activePhase_(FluidSystem::gasPhaseIdx);
        const auto [rs, rv] = clampRatio_(io, ig, a[Rs], a[Rv], e.wsRates);
        const auto [rsw, rvw] = clampRatio_(iw, ig, a[Rsw], a[Rvw], e.wsRates);
        const Scalar p = a[P], temp = a[T], salt = a[Salt];
        const int reg = e.pvtRegion;

        std::vector<Scalar> coeff(FluidSystem::numActivePhases(), 0.0);
        const Scalar detRw = 1.0 - rsw * rvw;
        if (iw >= 0) {
            const Scalar bw = FluidSystem::waterPvt()
                .inverseFormationVolumeFactor(reg, temp, p, rsw, salt);
            const Scalar den = bw * detRw;
            coeff[iw] += 1.0 / den;
            if (ig >= 0) {
                coeff[ig] -= rvw / den;
            }
        }
        const Scalar detR = 1.0 - rs * rv;
        if (io >= 0) {
            const Scalar bo = FluidSystem::oilPvt()
                .inverseFormationVolumeFactor(reg, temp, p, rs);
            const Scalar den = bo * detR;
            coeff[io] += 1.0 / den;
            if (ig >= 0) {
                coeff[ig] -= rv / den;
            }
        }
        if (ig >= 0) {
            const Scalar bg = FluidSystem::gasPvt()
                .inverseFormationVolumeFactor(reg, temp, p, rv, rvw);
            if (FluidSystem::enableDissolvedGasInWater()) {
                const Scalar denw = bg * detRw;
                coeff[ig] += 1.0 / denw;
                if (iw >= 0) {
                    coeff[iw] -= rsw / denw;
                }
            } else {
                const Scalar den = bg * detR;
                coeff[ig] += 1.0 / den;
                if (io >= 0) {
                    coeff[io] -= rs / den;
                }
            }
        }
        return coeff;
    }

    Scalar target_(const Entry& e, const AvgVector& a) const
    {
        if (!e.history) {
            return e.resvRate;
        }
        std::vector<Scalar> resv(FluidSystem::numActivePhases(), 0.0);
        converter_.calcReservoirVoidageRates(e.pvtRegion, a[P], a[Rs], a[Rv],
                                             a[Rsw], a[Rvw], a[T], a[Salt],
                                             e.hrates, resv);
        return std::accumulate(resv.begin(), resv.end(), Scalar{0});
    }

    //! R_ctrl at averages \p a with the converged rates held fixed
    //! (WellAssemble::assembleControlEqProd, case RESV).
    Scalar residual_(const Entry& e, const AvgVector& a) const
    {
        const auto coeff = coefficients_(e, a);
        Scalar total = 0.0;
        for (int p = 0; p < 3; ++p) {
            if (FluidSystem::phaseIsActive(p)) {
                total -= e.rates[p] * coeff[activePhase_(p)];
            }
        }
        return total - target_(e, a);
    }

    void countClampInactive_(const Entry& e, const AvgVector& a)
    {
        const int io = activePhase_(FluidSystem::oilPhaseIdx);
        const int ig = activePhase_(FluidSystem::gasPhaseIdx);
        if (io < 0 || ig < 0) {
            return;
        }
        const auto [rs, rv] = clampRatio_(io, ig, a[Rs], a[Rv], e.wsRates);
        rsClampInactive_ += (rs < a[Rs]) ? 1 : 0;
        rvClampInactive_ += (rv < a[Rv]) ? 1 : 0;
    }

    static bool nonNegative_(const int q)
    { return q != P && q != T; }

    bool varies_(const int q, const AvgVector& a) const
    {
        const bool oilGas = FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx) &&
                            FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx);
        switch (q) {
        case Rs:
            return oilGas && FluidSystem::enableDissolvedGas();
        case Rv:
            return oilGas && FluidSystem::enableVaporizedOil();
        case Rsw:
            return FluidSystem::enableDissolvedGasInWater();
        case Rvw:
            return FluidSystem::enableVaporizedWater();
        case Salt:
            return a[Salt] > 0.0;
        default:
            return true;
        }
    }

    static Scalar step_(const int q, const Scalar v)
    {
        switch (q) {
        case P:    return 1e-6 * std::max(std::abs(v), Scalar{1e5});
        case T:    return 1e-6 * std::max(std::abs(v), Scalar{1.0});
        case Rs:   return 1e-6 * std::max(std::abs(v), Scalar{1.0});
        case Salt: return 1e-6 * std::max(std::abs(v), Scalar{1e-3});
        default:   return 1e-6 * std::max(std::abs(v), Scalar{1e-4});
        }
    }

    //! Visit the interior cells with their AD quantities at time index 0,
    //! in the order (and with the expressions) of defineState.
    template<class Visitor>
    void forEachCell_(const Simulator& simulator, Visitor&& visit) const
    {
        ElementContext elemCtx(simulator);
        const auto& gridView = simulator.gridView();
        CellQuantities c;
        for (const auto& elem : elements(gridView, Dune::Partitions::interior)) {
            elemCtx.updatePrimaryStencil(elem);
            elemCtx.updatePrimaryIntensiveQuantities(/*timeIdx=*/0);
            const unsigned cellIdx = elemCtx.globalSpaceIndex(/*spaceIdx=*/0, /*timeIdx=*/0);
            const auto& iq = elemCtx.intensiveQuantities(/*spaceIdx=*/0, /*timeIdx=*/0);
            const auto& fs = iq.fluidState();
            const Scalar volume = simulator.model().dofTotalVolume(
                simulator.vanguard().gridEquilIdxToGridIdx(cellIdx));
            c.pv = volume * iq.porosity();
            c.hc = 1.0;
            if (FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)) {
                c.hc -= fs.saturation(FluidSystem::waterPhaseIdx);
            }
            const unsigned refPhase =
                FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx) ? FluidSystem::oilPhaseIdx
              : FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx) ? FluidSystem::gasPhaseIdx
                                                                     : FluidSystem::waterPhaseIdx;
            c.f[P] = fs.pressure(refPhase);
            c.f[T] = fs.temperature(refPhase);
            const bool oilGas = FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx) &&
                                FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx);
            c.f[Rs] = oilGas ? Evaluation(fs.Rs()) : Evaluation(0.0);
            c.f[Rv] = oilGas ? Evaluation(fs.Rv()) : Evaluation(0.0);
            c.so = oilGas ? Evaluation(fs.saturation(FluidSystem::oilPhaseIdx)) : Evaluation(0.0);
            c.sg = oilGas ? Evaluation(fs.saturation(FluidSystem::gasPhaseIdx)) : Evaluation(0.0);
            c.f[Rsw] = FluidSystem::enableDissolvedGasInWater() ? Evaluation(fs.Rsw())
                                                                : Evaluation(0.0);
            c.f[Rvw] = FluidSystem::enableVaporizedWater() ? Evaluation(fs.Rvw())
                                                           : Evaluation(0.0);
            c.f[Salt] = fs.saltConcentration();
            visit(cellIdx, c);
        }
    }

    bool enabled_;
    bool switchedOffByUser_;
    RateConverterType converter_;
    std::vector<Entry> entries_;
    // diagnostics
    long entriesApplied_{0};
    long substepsWithTerm_{0};
    long rsClampInactive_{0};
    long rvClampInactive_{0};
    Scalar maxCoeffMismatch_{0.0};
    Scalar maxTargetMismatch_{0.0};
    Scalar maxResidual_{0.0};
    AvgVector maxAbsG_{};
    long rowsCorrected_{0};
    Scalar maxRowCorrection_{0.0};
};

} // namespace Opm

#endif // OPM_ADJOINT_RESV_TERM_HPP
