//---------------------------------------------------------------------------//
/*
  Copyright (c) 2012, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the University of Wisconsin - Madison nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
//---------------------------------------------------------------------------//
/*!
 * \file MCLS_PRNG.hpp
 * \author Stuart R. Slattery
 * \brief Parallel random number generator class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_PRNG_HPP
#define MCLS_PRNG_HPP

#include <MCLS_config.hpp>
#include <MCLS_RNGTraits.hpp>

#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class PRNG
 * \brief Parallel manager class for c++11 random number generators.
 */
//---------------------------------------------------------------------------//
template<class RNG>
class PRNG
{
  public:

    //@{
    //! Typedefs.
    typedef RNG rng_type;
    typedef RNGTraits<RNG> RNGT;
    typedef typename RNGT::uniform_int_distribution_type IntDistribution;
    typedef RandomDistributionTraits<IntDistribution> IDT;
    //@}

    // Constructor.
    PRNG( const int comm_rank );

    // Destructor.
    ~PRNG()
    { /* ... */ }

    // Get a random number from a specified distribution.
    template<class RandomDistribution>
    inline typename RandomDistributionTraits<RandomDistribution>::result_type
    random( RandomDistribution& distribution );

  private:

    // Random number generator.
    Teuchos::RCP<RNG> d_rng;

#if HAVE_MCLS_TIMERS
    // Total random number generation timer.
    Teuchos::RCP<Teuchos::Time> d_rng_timer;
#endif
};

//---------------------------------------------------------------------------//
// Inline functions.
//---------------------------------------------------------------------------//
/*!
 * \brief Get a random number from a specified distribution.
 */
template<class RNG>
template<class RandomDistribution>
inline typename RandomDistributionTraits<RandomDistribution>::result_type
PRNG<RNG>::random( RandomDistribution& distribution )
{
#if HAVE_MCLS_TIMERS
    Teuchos::TimeMonitor rng_monitor( *d_rng_timer );
#endif
    return RNGT::random( *d_rng, distribution );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_PRNG_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_PRNG_HPP

//---------------------------------------------------------------------------//
// end MCLS_PRNG.hpp
//---------------------------------------------------------------------------//

