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
			    const int global_rank,
			    const Teuchos::RCP<Teuchos::ParameterList>& plist )
    : d_set_comm( set_comm )
    , d_plist( plist )
    , d_relative_weight_cutoff( 0.0 )
    , d_rng( Teuchos::rcp(new PRNG<rng_type>(global_rank)) )
#if HAVE_MCLS_TIMERS
    , d_mc_timer( Teuchos::TimeMonitor::getNewCounter("MCLS: Monte Carlo") )
#endif
{
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );
    MCLS_REQUIRE( Teuchos::nonnull(d_set_comm) );

    // Set the static byte size for the histories.
    HT::setByteSize();

    MCLS_ENSURE( HT::getPackedBytes() > 0 );
    MCLS_ENSURE( Teuchos::nonnull(d_rng) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Solve the linear problem. The domain and source must be set!
 */
template<class Source>
void MCSolver<Source>::solve()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_domain) );
    MCLS_REQUIRE( Teuchos::nonnull(d_source) );
    MCLS_REQUIRE( Teuchos::nonnull(d_tally) );
    MCLS_REQUIRE( Teuchos::nonnull(d_transporter) );
    
    // Zero out the tally.
    TT::zeroOut( *d_tally );

    // Assign the source to the transporter.
    d_transporter->assignSource( d_source, d_relative_weight_cutoff );

    // Transport the source to solve the problem.
    {
#if HAVE_MCLS_TIMERS
	Teuchos::TimeMonitor mc_monitor( *d_mc_timer );
#endif
	d_transporter->transport();
    }

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
    MCLS_REQUIRE( Teuchos::nonnull(domain) );

    // Set the domain.
    d_domain = domain;

    // Set the random number generator with the domain.
    DT::setRNG( *d_domain, d_rng );

    // Get the domain tally.
    d_tally = DT::domainTally( *d_domain );

    // Generate the source transporter.
    d_transporter = GlobalTransporterFactory<Source>::create(
        d_set_comm, d_domain, *d_plist );

    MCLS_ENSURE( Teuchos::nonnull(d_domain) );
    MCLS_ENSURE( Teuchos::nonnull(d_tally) );
    MCLS_ENSURE( Teuchos::nonnull(d_transporter) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the source for transport. Must always be called after
 * setDomain(). 
 */
template<class Source>
void MCSolver<Source>::setSource( const Teuchos::RCP<Source>& source )
{
    MCLS_REQUIRE( Teuchos::nonnull(source) );

    // Set the source.
    d_source = source;

    // Set the random number generator with the source.
    ST::setRNG( *d_source, d_rng );

    // Build the source.
    ST::buildSource( *d_source );

    // Get the weight cutoff.
    double cutoff = d_plist->get<double>("Weight Cutoff");
    d_relative_weight_cutoff = cutoff * ST::weight( *d_source, 0 );

    MCLS_ENSURE( Teuchos::nonnull(d_source) );
    MCLS_ENSURE( d_relative_weight_cutoff > 0.0 );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_MCSOLVER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_MCSolver_impl.hpp
// ---------------------------------------------------------------------------//

