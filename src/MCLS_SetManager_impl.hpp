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
 * \file MCLS_SetManager_impl.hpp
 * \author Stuart R. Slattery
 * \brief Multiple set manager implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SETMANAGER_IMPL_HPP
#define MCLS_SETMANAGER_IMPL_HPP

#include "MCLS_DBC.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_as.hpp>
#include <Teuchos_Ptr.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector, class Matrix>
SetManager<Vector,Matrix>::SetManager( 
    const Teuchos::RCP<LinearProblemType>& primary_problem,
    const Teuchos::RCP<const Comm>& global_comm,
    Teuchos::ParameterList& plist )
    : d_global_comm( global_comm )
    , d_num_sets( plist.get<int>("Number of Sets") )
    , d_set_size( 0 )
    , d_set_id( 0 )
    , d_problems( d_num_sets )
    , d_set_comms( d_num_sets )
    , d_p_to_s_exports( d_num_sets - 1 )
    , d_s_to_p_exports( d_num_sets - 1 )
{
    Require( !primary_problem.is_null() );
    Require( !global_comm.is_null() );
    Require( d_num_sets > 0 );

    // Get the set size. We could compute this value from user input, but we
    // must Insist that this is true every time and therefore we do this
    // reduction to verify. We require the primary problem to be have no data
    // on procs not owned by the primary linear problem.
    int local_size = Teuchos::as<int>( 
	(VT::getLocalLength(*primary_problem->getLHS()) > 0) ); 
    Teuchos::reduceAll<int,int>( *d_global_comm, Teuchos::REDUCE_SUM,
				 local_size, Teuchos::Ptr<int>(&d_set_size) );
    Insist( d_num_sets * d_set_size == d_global_comm->getSize(),
	    "Size of set * Number of sets != Global communicator size!" );
    Check( d_set_size > 0 );

    // We require that the primary problem exist on global procs 0 through
    // (d_set_size-1).
    if ( d_global_comm->getRank() < d_set_size )
    {
	Insist(VT::getLocalLength(*primary_problem->getLHS()) > 0,
	       "Primary linear problem must exist on procs [0,(set_size-1)] only!");
    }
    else
    {
	Insist(VT::getLocalLength(*primary_problem->getLHS()) == 0,
	       "Primary linear problem must exist on procs [0,(set_size-1)] only!");
    }

    // Add the primary set to the linear problem array.
    d_problems[0] = primary_problem;

    // Generate the set-constant communicator for the primary problem.
    Teuchos::Array<int> subcomm_ranks( d_set_size );
    for ( int n = 0; n < d_set_size; ++n )
    {
	subcomm_ranks[n] = n;
    }
    d_set_comms[0] = d_global_comm->createSubcommunicator( subcomm_ranks() );

    // Build the secondary linear problems.
    for ( int p = 1; p < d_num_sets; ++p )
    {
	
    }
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_SETMANAGER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_SetManager_impl.hpp
// ---------------------------------------------------------------------------//

