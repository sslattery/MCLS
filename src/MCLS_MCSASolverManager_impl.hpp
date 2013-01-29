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
 * \file MCLS_MCSASolverManager_impl.hpp
 * \author Stuart R. Slattery
 * \brief Monte Carlo Synthetic Acceleration solver manager implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MCSASOLVERMANAGER_IMPL_HPP
#define MCLS_MCSASOLVERMANAGER_IMPL_HPP

#include <string>

#include "MCLS_DBC.hpp"
#include "MCLS_AdjointSolverManager.hpp"

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector, class Matrix>
MCSASolverManager<Vector,Matrix>::MCSASolverManager( 
    const Teuchos::RCP<LinearProblemType>& problem,
    const Teuchos::RCP<const Comm>& global_comm,
    const Teuchos::RCP<Teuchos::ParameterList>& plist )
    : d_problem( problem )
    , d_global_comm( global_comm )
    , d_plist( plist )
    , d_primary_set( !d_problem.is_null() )
    , d_num_iters( 0 )
    , d_converged_status( false )
{
    // Generate the residual Monte Carlo problem on the primary set.
    Teuchos::RCP<LinearProblemType> residual_problem;
    if ( d_primary_set )
    {
	Teuchos::RCP<VectorType> delta_x = VT::clone( d_problem->getLHS() );
	residual_problem = Teuchos::rcp(
	    new LinearProblemType( d_problem->getOperator(),
				   delta_x,
				   d_problem->getResidual() ) );
    }

    // Create the Monte Carlo direct solver.
    if ( d_plist->get<std::string>("MC Type") == "Adjoint" )
    {
	d_direct_solver = Teuchos::rcp( 
	    new AdjointSolverManager(residual_problem, global_comm, plist) );
    }

    Ensure( !d_direct_solver.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the valid parameters for this manager.
 */
template<class Vector, class Matrix>
Teuchos::RCP<const Teuchos::ParameterList> 
MCSASolverManager<Vector,Matrix>::getValidParameters() const
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the tolerance achieved on the last linear solve. This may be
 * less or more than the set convergence tolerance. 
 */
template<class Vector, class Matrix>
typename Teuchos::ScalarTraits<
    typename MCSASolverManager<Vector,Matrix>::Scalar>::magnitudeType 
MCSASolverManager<Vector,Matrix>::achievedTol() const
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the linear problem with the manager.
 */
template<class Vector, class Matrix>
void MCSASolverManager<Vector,Matrix>::setProblem( 
    const Teuchos::RCP<LinearProblem<Vector,Matrix> >& problem )
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the parameters for the manager. The manager will modify this
 * list with default parameters that are not defined.
 */
template<class Vector, class Matrix>
void MCSASolverManager<Vector,Matrix>::setParameters( 
    const Teuchos::RCP<Teuchos::ParameterList>& params )
{
    Require( !params.is_null() );
    d_plist = params;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Solve the linear problem. Return true if the solution
 * converged. False if it did not.
 */
template<class Vector, class Matrix>
bool MCSASolverManager<Vector,Matrix>::solve()
{

}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_MCSASOLVERMANAGER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_MCSASolverManager_impl.hpp
//---------------------------------------------------------------------------//

