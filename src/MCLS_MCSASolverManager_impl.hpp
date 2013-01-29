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
    if ( d_primary_set )
    {
	Teuchos::RCP<Vector> delta_x = VT::clone( *d_problem->getLHS() );
	d_residual_problem = Teuchos::rcp(
	    new LinearProblemType( d_problem->getOperator(),
				   delta_x,
				   d_problem->getResidual() ) );
    }
    d_global_comm->barrier();

    // Create the Monte Carlo direct solver for the residual problem.
    if ( d_plist->get<std::string>("MC Type") == "Adjoint" )
    {
	d_mc_solver = Teuchos::rcp( 
	    new AdjointSolverManager<Vector,Matrix>(
		d_residual_problem, global_comm, plist) );
    }

    Ensure( !d_mc_solver.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the valid parameters for this manager.
 */
template<class Vector, class Matrix>
Teuchos::RCP<const Teuchos::ParameterList> 
MCSASolverManager<Vector,Matrix>::getValidParameters() const
{
    // Create a parameter list with the Monte Carlo solver parameters as a
    // starting point.
    Teuchos::RCP<Teuchos::ParameterList> plist = 
	Teuchos::parameterList( *d_mc_solver->getValidParameters() );

    // Add the default code values. Put zero if no default.
    plist->set<std::string>("MC Type", "Adjoint");
    plist->set<double>("Convergence Tolerance", 0.0);
    plist->set<int>("Max Number of Iterations", 0);

    return plist;
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
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType residual_norm = 
	Teuchos::ScalarTraits<Scalar>::zero();

    // We only do this on the primary set where the linear problem exists.
    if ( d_primary_set )
    {
	residual_norm = VT::normInf( *d_problem->getResidual() );
	residual_norm /= VT::normInf( *d_problem->getRHS() );
    }
    d_global_comm->barrier();

    return residual_norm;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the linear problem with the manager.
 */
template<class Vector, class Matrix>
void MCSASolverManager<Vector,Matrix>::setProblem( 
    const Teuchos::RCP<LinearProblem<Vector,Matrix> >& problem )
{
    Require( !d_global_comm.is_null() );
    Require( !d_plist.is_null() );

    // Set the MCSA problem.
    d_problem = problem;
    d_primary_set = !d_problem.is_null();

    // Update the residual problem.
    if ( d_primary_set )
    {
	d_residual_problem->setOperator( d_problem->getOperator() );
	d_residual_problem->setRHS( d_problem->getResidual() );
    }
    d_global_comm->barrier();

    // Set the updated residual problem with the Monte Carlo solver.
    d_mc_solver->setProblem( d_residual_problem );
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

    // Propagate the parameters to the Monte Carlo solver.
    d_mc_solver->setParameters( d_plist );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Solve the linear problem. Return true if the solution
 * converged. False if it did not.
 */
template<class Vector, class Matrix>
bool MCSASolverManager<Vector,Matrix>::solve()
{
    // Convergence parameters.
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType tolerance = 
	d_plist->get<double>("Convergence Tolerance");
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType source_norm =
	VT::normInf( *d_problem->getRHS() );
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType 
	convergence_criteria = tolerance * source_norm;
    d_converged_status = false;

    // Iteration setup.
    int max_num_iters = d_plist->get<int>("Max Number of Iterations");
    d_num_iters = 0;
    int print_freq = 10;

    // Compute the initial residual.
    d_problem->updateResidual();
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType residual_norm =
	VT::normInf( *d_problem->getResidual() );

    // Temporary vector for Richardson iteration.
    Teuchos::RCP<Vector> tmp = VT::clone( *d_problem->getLHS() );	

    // Iterate.
    while( residual_norm > convergence_criteria &&
	   d_num_iters < max_num_iters )
    {
	// Do a Richardson iteration.
	d_problem->applyOperator( *d_problem->getLHS(), *tmp );
	VT::update( 
	    *d_problem->getLHS(), Teuchos::ScalarTraits<Scalar>::one(),
	    *tmp, -Teuchos::ScalarTraits<Scalar>::one(),
	    *d_problem->getRHS(), Teuchos::ScalarTraits<Scalar>::one() );

	// Update the residual.
	d_problem->updateResidual();

	// Solve the residual Monte Carlo problem.
	d_mc_solver->solve();

	// Apply the correction.
	VT::update( *d_problem->getLHS(), 
		    Teuchos::ScalarTraits<Scalar>::one(),
		    *d_residual_problem->getLHS(), 
		    Teuchos::ScalarTraits<Scalar>::one() );

	// Update the residual.
	d_problem->updateResidual();
	residual_norm = VT::normInf( *d_problem->getResidual() );

	// Update the iteration count.
	++d_num_iters;

	// Print iteration data.
	if ( d_global_comm->getRank() == 0 && d_num_iters % print_freq == 0 )
	{
	    std::cout << "MCSA Iteration " << d_num_iters 
		      << ": Residual = " 
		      << residual_norm/source_norm << std::endl;
	}

	// Barrier before proceeding.
	d_global_comm->barrier();
    }

    // Check for convergence.
    if ( VT::normInf(*d_problem->getResidual()) <= convergence_criteria )
    {
	d_converged_status = true;
    }

    return d_converged_status;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_MCSASOLVERMANAGER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_MCSASolverManager_impl.hpp
//---------------------------------------------------------------------------//

