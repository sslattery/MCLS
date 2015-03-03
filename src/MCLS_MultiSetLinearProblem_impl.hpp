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
 * \file MCLS_MultiSetLinearProblem_impl.hpp
 * \author Stuart R. Slattery
 * \brief Linear problem implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MULTISETLINEARPROBLEM_IMPL_HPP
#define MCLS_MULTISETLINEARPROBLEM_IMPL_HPP

#include "MCLS_DBC.hpp"

#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_CommHelpers.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Linear problem onstructor.
 */
template<class Vector, class Matrix>
MultiSetLinearProblem<Vector,Matrix>::MultiSetLinearProblem( 
    const Teuchos::RCP<const Teuchos::Comm<int> >& global_comm,
    const int num_sets,
    const int set_id,
    const Teuchos::RCP<LinearProblem<Vector,Matrix> >& problem )
    : d_global_comm( global_comm )
    , d_num_sets( num_sets )
    , d_set_id( set_id )
    , d_problem( problem )
#if HAVE_MCLS_TIMERS
    , d_bcvs_timer(
	Teuchos::TimeMonitor::getNewCounter("MCLS: Block-Constant Sum") )
#endif
{
    buildCommunicators();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Linear problem onstructor.
 */
template<class Vector, class Matrix>
MultiSetLinearProblem<Vector,Matrix>::MultiSetLinearProblem( 
    const Teuchos::RCP<const Teuchos::Comm<int> >& global_comm,
    const int num_sets,
    const int set_id,
    const Teuchos::RCP<const Matrix>& A,
    const Teuchos::RCP<Vector>& x,
    const Teuchos::RCP<const Vector>& b )
    : d_global_comm( global_comm )
    , d_num_sets( num_sets )
    , d_set_id( set_id )
    , d_problem( linearProblem(A,x,b) )
#if HAVE_MCLS_TIMERS
    , d_bcvs_timer(
	Teuchos::TimeMonitor::getNewCounter("MCLS: Block-Constant Sum") )
#endif
{
    buildCommunicators();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the linear operator.
 */
template<class Vector, class Matrix>
void MultiSetLinearProblem<Vector,Matrix>::blockConstantVectorSum( 
    const Teuchos::RCP<Vector>& vector ) const
{
#if HAVE_MCLS_TIMERS
    Teuchos::TimeMonitor bcvs_monitor( *d_bcvs_timer );
#endif

    MCLS_REQUIRE( Teuchos::nonnull(vector) );

    Teuchos::ArrayRCP<Scalar> vector_view = VT::viewNonConst( *vector );
    Teuchos::ArrayRCP<Scalar> vector_copy;
    vector_copy.assign( vector_view.begin(), vector_view.end() );
    Teuchos::reduceAll( *d_block_comm,
			Teuchos::REDUCE_SUM,
			vector_view.size(),
			vector_copy.getRawPtr(),
			vector_view.getRawPtr() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the linear operator.
 */
template<class Vector, class Matrix>
void MultiSetLinearProblem<Vector,Matrix>::buildCommunicators()
{
    // Get the set-constant communicator.
    d_set_comm = VT::getComm( *d_problem->getLHS() );

    // The block id is the rank in the set communicator.
    d_block_id = d_set_comm->getRank();

    // Split the global communicator to make the block-constant
    // communicators.
    d_block_comm = d_global_comm->split( d_block_id, d_set_id );

    MCLS_ENSURE( d_set_comm->getSize()*d_num_sets == d_global_comm->getSize() );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_MULTISETLINEARPROBLEM_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_MultiSetLinearProblem_impl.hpp
//---------------------------------------------------------------------------//

