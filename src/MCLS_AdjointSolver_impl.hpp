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
 * \file MCLS_AdjointSolver_impl.hpp
 * \author Stuart R. Slattery
 * \brief Adjoint Monte Carlo solver implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ADJOINTSOLVER_IMPL_HPP
#define MCLS_ADJOINTSOLVER_IMPL_HPP

#include <string>

#include "MCLS_DBC.hpp"
#include "MCLS_UniformAdjointSource.hpp"

#include <Teuchos_ScalarTraits.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector, class Matrix>
AdjointSolver<Vector,Matrix>::AdjointSolver( 
    const Teuchos::RCP<LinearProblemType>& linear_problem,
    Teuchos::ParameterList& plist,
    int seed )
    : d_linear_problem( linear_problem )
    , d_seed( seed )
    , d_set_comm( MT::getComm(*d_linear_problem->getOperator()) )
{
    Require( !d_linear_problem.is_null() );
    Require( d_linear_problem->status() );

    // Check for a user provided random number seed. The default is provided
    // as a default argument for this constructor.
    if ( plist.isParam("Random Number Seed") )
    {
	d_seed = plist.get<int>("Random Number Seed");
    }

    // Build the random number generator.
    d_rng_control = Teuchos::rcp( new RNGControl(seed) );

    // Set the static byte size for the histories. If we want reproducible
    // results we pack the RNG with the histories. If we don't, then we use
    // the global RNG.
    if ( plist.get<bool>("Reproducible MC Mode") )
    {
	HistoryType::setByteSize( d_rng_control->getSize() );
    }
    else
    {
	HistoryType::setByteSize( 0 );
    }

    // Generate the domain.
    d_domain = Teuchos::rcp( new DomainType( d_linear_problem->getOperator(),
					     d_linear_problem->getLHS(),
					     plist ) );

    // Get the domain tally.
    d_tally = d_domain->domainTally();

    // Generate the initial source.
    setSource();

    // Generate the source transporter.
    d_transporter = 
	Teuchos::rcp( new TransporterType(set_comm, d_domain, plist) );

    Ensure( HistoryType::getPackedBytes() > 0 );
    Ensure( !d_set_comm.is_null() );
    Ensure( !d_rng_control.is_null() );
    Ensure( !d_domain.is_null() );
    Ensure( !d_tally.is_null() );
    Ensure( !d_source.is_null() );
    Ensure( !d_transporter.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Solve the linear problem.
 */
template<class Vector, class Matrix>
void AdjointSolver<Vector,Matrix>::solve()
{
    // Zero out the LHS.
    VT::putScalar( *d_linear_problem->getLHS(), 
		   Teuchos::ScalarTraits<typename VT::scalar_type>::zero() );

    // If the RHS of the linear problem has changed since the last solve,
    // update the source.
    if ( !d_linear_problem->status() )
    {
	setSource();
	d_linear_problem->setProblem();
    }
    Check( d_linear_problem->status() );

    // Assign the source to the transporter.
    d_transporter->assignSource( d_source );

    // Transport the source to solve the problem.
    d_transporter.transport();

    // Barrier after completion.
    d_set_comm->barrier();

    // Update the set tallies.
    d_tally->combineTallies();

    // Normalize the tally with the number of source histories in the set.
    d_tally->normalize( d_source->numToTransportInSet() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the source based on the RHS of the linear problem.
 */
template<class Vector, class Matrix>
void AdjointSolver<Vector,Matrix>::setSource()
{
    // Initialize source.
    d_source = Teuchos::rcp( new UniformAdjointSource<DomainType>(
				 d_linear_problem->getRHS(),
				 d_domain,
				 d_rng_control,
				 d_set_comm,
				 plist ) );

    // Generate the source.
    source->buildSource();

    Ensure( !d_source.is_null() );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_ADJOINTSOLVER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_AdjointSolver_impl.hpp
// ---------------------------------------------------------------------------//

