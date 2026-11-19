/*
  Copyright 2026 SINTEF Digital, Mathematics and Cybernetics.

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
 * \brief WGState normalization for the adjoint recorder, against an
 *        unmodified opm-simulators well model.
 *
 * The pre-step snapshot of report step 0 is only deserializable against a
 * prepareDeserialize()'d well model if the internal WGState copies
 * (last_valid, nupcol) are synchronized with the active one.
 * commitWGState() is public upstream, but updateNupcolWGState() is
 * protected; it is reached here through the standard member-pointer
 * access pattern (forming the pointer through a derived class is allowed
 * and it can then be invoked on any base object). Both copies would be
 * assigned the same values during the first iteration of the step anyway.
 *
 * (A derived well model installed via the WellModel property would be the
 * cleaner route, but NonlinearSystemBlackOilReservoir's constructor
 * currently takes the concrete BlackoilWellModel<TypeTag>&, so a property
 * override does not propagate.)
 */
#ifndef OPM_ADJOINT_WELL_MODEL_HPP
#define OPM_ADJOINT_WELL_MODEL_HPP

#include <opm/simulators/wells/BlackoilWellModelGeneric.hpp>

namespace Opm {

namespace detail {

template<class Scalar, class IndexTraits>
class WGStateNormalizer : public BlackoilWellModelGeneric<Scalar, IndexTraits>
{
public:
    // Never instantiated; only used to form the protected member pointer.
    WGStateNormalizer() = delete;

    template<class WellModel>
    static void normalize(WellModel& wellModel)
    {
        wellModel.commitWGState();
        constexpr auto updateNupcol = &WGStateNormalizer::updateNupcolWGState;
        (wellModel.*updateNupcol)();
    }

    template<class WellModel>
    static void clearDeserializedStatePending(WellModel& wellModel)
    {
        constexpr auto pending = &WGStateNormalizer::deserialized_state_pending_;
        wellModel.*pending = false;
    }
};

} // namespace detail

//! \brief Let the next beginReportStep() initialise the well state again.
//!
//! Since OPM/opm-simulators#7405 (2026-08-28), prepareDeserialize() marks the
//! well state as freshly deserialized, and the next beginReportStep() skips
//! initializeWellState() and keeps the restored state as it is. That is right
//! for a restart, where the snapshot belongs to the report step being started.
//! In the adjoint replay the snapshot is the state at the end of substep k-1,
//! possibly in the previous report step, so a well that opens at report step k
//! would be missing from the well state (out_of_range from
//! GlobalWellInfo::in_producing_group in beginTimeStep). Clearing the flag
//! after the restore keeps the behaviour the replay was written against.
template<class WellModel>
void clearDeserializedStatePending(WellModel& wellModel)
{
    detail::WGStateNormalizer<typename WellModel::Scalar,
                              typename WellModel::IndexTraits>::clearDeserializedStatePending(wellModel);
}

//! \brief Synchronize the internal WGState copies (last_valid, nupcol)
//!        with the active one so a pre-step snapshot can later be
//!        deserialized against a prepareDeserialize()'d well model.
template<class WellModel>
void normalizeWGStateForSerialization(WellModel& wellModel)
{
    detail::WGStateNormalizer<typename WellModel::Scalar,
                              typename WellModel::IndexTraits>::normalize(wellModel);
}

} // namespace Opm

#endif // OPM_ADJOINT_WELL_MODEL_HPP
