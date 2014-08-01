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
 * \file MCLS_PRNG_impl.hpp
 * \author Stuart R. Slattery
 * \brief Parallel random number generator class implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_PRNG_IMPL_HPP
#define MCLS_PRNG_IMPL_HPP

#include <random>
#include <limits>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 *
 * \param comm_rank Rank of the local process in the global parallel
 * communicator.
 */
template<class RNG>
PRNG<RNG>::PRNG( const int comm_rank)
{
    // Create a random device to get an initial random number. This is
    // potentially non-deterministic.
    std::random_device rand_device;

    // Create a master rng to produce seed values for each parallel rank.
    Teuchos::RCP<RNG> master_rng = RNGT::create( rand_device() );
    typename RNGT::uniform_int_distribution_type::result_type seed = 0;
    Teuchos::RCP<typename RNGT::uniform_int_distribution_type> distribution =
	RandomDistributionTraits<RNGT::uniform_int_distribution_type>::create(
	    0, std::numeric_limits<
	    typename RNGT::uniform_int_distribution_type::result_type>::max() );
    for ( int i = 0; i < comm_rank; ++i )
    {
	seed = RNGT::random( *master_rng, *distribution );
    }

    // Seed the random number generator on this process.
    d_rng = RNGT::create( seed );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_PRNG_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_PRNG_impl.hpp
//---------------------------------------------------------------------------//

