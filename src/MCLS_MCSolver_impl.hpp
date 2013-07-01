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
 * \file MCLS_MCSolver_impl.hpp
 * \author Stuart R. Slattery
 * \brief Monte Carlo solver implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MCSOLVER_IMPL_HPP
#define MCLS_MCSOLVER_IMPL_HPP

#include <string>

#include "MCLS_DBC.hpp"
#include "MCLS_GlobalTransporterFactory.hpp"

#include <Teuchos_ScalarTraits.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Source>
MCSolver<Source>::MCSolver( const Teuchos::RCP<const Comm>& set_comm,
			    const Teuchos::RCP<Teuchos::ParameterList>& plist,
			    int seed )
    : d_set_comm( set_comm )
    , d_plist( plist )
    , d_relative_weight_cutoff( 0.0 )
{
    MCLS_REQUIRE( !d_plist.is_null() );
    MCLS_REQUIRE( !d_set_comm.is_null() );

    // Check for a user provided random number seed. The default is provided
    // as a default argument for this constructor.
    if ( d_plist->isParameter("Random Number Seed") )
    {
	seed = d_plist->get<int>("Random Number Seed");
    }

    // Build the random number generator.
    d_rng_control = Teuchos::rcp( new RNGControl(seed) );

    // Set the static byte size for the histories. If we want reproducible
    // results we pack the RNG with the histories. If we don't, then we use
    // the global RNG.
    if ( d_plist->get<bool>("Reproducible MC Mode") )
    {
	HT::setByteSize( d_rng_control->getSize() );
    }
    else
    {
	HT::setByteSize( 0 );
    }

    MCLS_ENSURE( HT::getPackedBytes() > 0 );
    MCLS_ENSURE( !d_rng_control.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Solve the linear problem. The domain and source must be set!
 */
template<class Source>
void MCSolver<Source>::solve()
{
    MCLS_REQUIRE( !d_domain.is_null() );
    MCLS_REQUIRE( !d_source.is_null() );
    MCLS_REQUIRE( !d_tally.is_null() );
    MCLS_REQUIRE( !d_transporter.is_null() );
    
    // Zero out the tally.
    TT::zeroOut( *d_tally );

    // Assign the source to the transporter.
    d_transporter->assignSource( d_source, d_relative_weight_cutoff );

    // Transport the source to solve the problem.
    d_transporter->transport();

    // Update the set tallies.
    TT::combineSetTallies( *d_tally, d_set_comm );

    // Normalize the tally with the number of source histories in the set.
    TT::normalize( *d_tally, ST::numToTransportInSet(*d_source) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the domain for transport.
 */
template<class Source>
void MCSolver<Source>::setDomain( const Teuchos::RCP<Domain>& domain )
{
    MCLS_REQUIRE( !domain.is_null() );

    // Set the domain.
    d_domain = domain;

    // Get the domain tally.
    d_tally = DT::domainTally( *d_domain );

    // Generate the source transporter.
    d_transporter = GlobalTransporterFactory<Source>::create(
        d_set_comm, d_domain, *d_plist );

    MCLS_ENSURE( !d_domain.is_null() );
    MCLS_ENSURE( !d_tally.is_null() );
    MCLS_ENSURE( !d_transporter.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the source for transport. Must always be called after
 * setDomain(). 
 */
template<class Source>
void MCSolver<Source>::setSource( const Teuchos::RCP<Source>& source )
{
    MCLS_REQUIRE( !source.is_null() );

    d_source = source;
    ST::buildSource( *d_source );

    double cutoff = d_plist->get<double>("Weight Cutoff");
    d_relative_weight_cutoff = cutoff * ST::weight( *d_source, 0 );

    MCLS_ENSURE( !d_source.is_null() );
    MCLS_ENSURE( d_relative_weight_cutoff > 0.0 );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_MCSOLVER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_MCSolver_impl.hpp
// ---------------------------------------------------------------------------//

