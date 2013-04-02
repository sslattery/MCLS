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
 * \file MCLS_SequentialMCSolverManager_impl.hpp
 * \author Stuart R. Slattery
 * \brief Sequential Monte Carlo solver manager implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SEQUENTIALMCSOLVERMANAGER_IMPL_HPP
#define MCLS_SEQUENTIALMCSOLVERMANAGER_IMPL_HPP

#include <string>

#include "MCLS_DBC.hpp"
#include "MCLS_AdjointSolverManager.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_Ptr.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Comm constructor. setProblem() must be called before solve().
 */
template<class Vector, class Matrix>
SequentialMCSolverManager<Vector,Matrix>::SequentialMCSolverManager( 
    const Teuchos::RCP<const Comm>& global_comm,
    const Teuchos::RCP<Teuchos::ParameterList>& plist )
    : d_global_comm( global_comm )
    , d_plist( plist )
{
    MCLS_REQUIRE( !d_global_comm.is_null() );
    MCLS_REQUIRE( !d_plist.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector, class Matrix>
SequentialMCSolverManager<Vector,Matrix>::SequentialMCSolverManager( 
    const Teuchos::RCP<LinearProblemType>& problem,
    const Teuchos::RCP<const Comm>& global_comm,
    const Teuchos::RCP<Teuchos::ParameterList>& plist )
    : d_problem( problem )
    , d_global_comm( global_comm )
    , d_plist( plist )
    , d_primary_set( !d_problem.is_null() )
    , d_num_iters( 0 )
    , d_converged_status( 0 )
{
    MCLS_REQUIRE( !global_comm.is_null() );
    MCLS_REQUIRE( !d_plist.is_null() );

    buildResidualMonteCarloProblem();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the valid parameters for this manager.
 */
template<class Vector, class Matrix>
Teuchos::RCP<const Teuchos::ParameterList> 
SequentialMCSolverManager<Vector,Matrix>::getValidParameters() const
{
    // Create a parameter list with the Monte Carlo solver parameters as a
    // starting point.
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    if ( Teuchos::nonnull(d_mc_solver) )
    {
	plist->setParameters( *d_mc_solver->getValidParameters() );
    }

    // Add the default code values. Put zero if no default.
    plist->set<std::string>("MC Type", "Adjoint");
    plist->set<double>("Convergence Tolerance", 1.0);
    plist->set<int>("Maximum Iterations", 1000);
    plist->set<int>("Iteration Print Frequency", 10);
    plist->set<int>("Iteration Check Frequency", 1);

    return plist;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the tolerance achieved on the last linear solve. This may be
 * less or more than the set convergence tolerance. 
 */
template<class Vector, class Matrix>
typename Teuchos::ScalarTraits<
    typename SequentialMCSolverManager<Vector,Matrix>::Scalar>::magnitudeType 
SequentialMCSolverManager<Vector,Matrix>::achievedTol() const
{
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType residual_norm = 
	Teuchos::ScalarTraits<Scalar>::zero();

    // We only do this on the primary set where the linear problem exists.
    if ( d_primary_set )
    {
	typename Teuchos::ScalarTraits<Scalar>::magnitudeType source_norm = 0;

	residual_norm = VT::normInf( *d_problem->getPrecResidual() );

	// Compute the source norm preconditioned if necessary.
	if ( d_problem->isLeftPrec() )
	{
	    Teuchos::RCP<Vector> tmp = VT::clone( *d_problem->getRHS() );
	    d_problem->applyLeftPrec( *d_problem->getRHS(), *tmp );
	    source_norm = VT::normInf( *tmp );
	}
	else
	{
	    source_norm = VT::normInf( *d_problem->getRHS() );
	}

	residual_norm /= source_norm;
    }
    d_global_comm->barrier();

    return residual_norm;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the linear problem with the manager.
 */
template<class Vector, class Matrix>
void SequentialMCSolverManager<Vector,Matrix>::setProblem( 
    const Teuchos::RCP<LinearProblem<Vector,Matrix> >& problem )
{
    MCLS_REQUIRE( !d_global_comm.is_null() );
    MCLS_REQUIRE( !d_plist.is_null() );

    // Set the problem.
    d_problem = problem;
    d_primary_set = !d_problem.is_null();

    // Update the residual problem is it already exists.
    if ( Teuchos::nonnull(d_mc_solver) )
    {
	if ( d_primary_set )
	{
	    d_residual_problem->setOperator( d_problem->getCompositeOperator() );
	    d_residual_problem->setRHS( d_problem->getPrecResidual() );
	}
	d_global_comm->barrier();

	// Set the updated residual problem with the Monte Carlo solver.
	d_mc_solver->setProblem( d_residual_problem );
    }

    // Otherwise this is initialization.
    else
    {
	buildResidualMonteCarloProblem();
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the parameters for the manager. The manager will modify this
 * list with default parameters that are not defined.
 */
template<class Vector, class Matrix>
void SequentialMCSolverManager<Vector,Matrix>::setParameters( 
    const Teuchos::RCP<Teuchos::ParameterList>& params )
{
    MCLS_REQUIRE( Teuchos::nonnull(params) );
    MCLS_REQUIRE( Teuchos::nonnull(d_mc_solver) );

    // Set the parameters.
    d_plist = params;

    // Propagate the parameters to the existing Monte Carlo solver.
    d_mc_solver->setParameters( d_plist );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Solve the linear problem. Return true if the solution
 * converged. False if it did not.
 */
template<class Vector, class Matrix>
bool SequentialMCSolverManager<Vector,Matrix>::solve()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_global_comm) );
    MCLS_REQUIRE( Teuchos::nonnull(d_mc_solver) );
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );

    // Get the convergence parameters on the primary set.
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType 
	convergence_criteria = 0;
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType source_norm = 0;
    if ( d_primary_set )
    {
        typename Teuchos::ScalarTraits<double>::magnitudeType tolerance = 1.0;
        if ( d_plist->isParameter("Convergence Tolerance") )
        {
            tolerance = d_plist->get<double>("Convergence Tolerance");
        }

	// Compute the source norm preconditioned if necessary.
	if ( d_problem->isLeftPrec() )
	{
	    Teuchos::RCP<Vector> tmp = VT::clone( *d_problem->getRHS() );
	    d_problem->applyLeftPrec( *d_problem->getRHS(), *tmp );
	    source_norm = VT::normInf( *tmp );
	}
	else
	{
	    source_norm = VT::normInf( *d_problem->getRHS() );
	}
	
	convergence_criteria = tolerance * source_norm;
    }
    d_global_comm->barrier();
    d_converged_status = 0;

    // Iteration setup.
    int max_num_iters = 1000;
    if ( d_plist->isParameter("Maximum Iterations") )
    {
        max_num_iters = d_plist->get<int>("Maximum Iterations");
    }
    d_num_iters = 0;
    int print_freq = 10;
    if ( d_plist->isParameter("Iteration Print Frequency") )
    {
	print_freq = d_plist->get<int>("Iteration Print Frequency");
    }
    int check_freq = 1;
    if ( d_plist->isParameter("Iteration Check Frequency") )
    {
	check_freq = d_plist->get<int>("Iteration Check Frequency");
    }

    // Setup for iteration.
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType residual_norm = 0;
    if ( d_primary_set )
    {
	// Compute the initial preconditioned residual.
	d_problem->updatePrecResidual();
	residual_norm = VT::normInf( *d_problem->getPrecResidual() );
    }
    d_global_comm->barrier();

    // Iterate.
    int do_iterations = 1;
    while( do_iterations )
    {
	// Update the iteration count.
	++d_num_iters;

	// Solve the residual Monte Carlo problem.
	d_mc_solver->solve();

	// Apply the correction and update the preconditioned residual on the
	// primary set.
	if ( d_primary_set )
	{
	    VT::update( *d_problem->getLHS(), 
			Teuchos::ScalarTraits<Scalar>::one(),
			*d_residual_problem->getLHS(), 
			Teuchos::ScalarTraits<Scalar>::one() );

	    d_problem->updatePrecResidual();
	    residual_norm = VT::normInf( *d_problem->getPrecResidual() );

	    // Check if we're done iterating.
	    if ( d_num_iters % check_freq == 0 )
	    {
		do_iterations = (residual_norm > convergence_criteria) &&
				(d_num_iters < max_num_iters);
	    }
	}
	d_global_comm->barrier();

	// Broadcast iteration status to the blocks.
	if ( d_num_iters % check_freq == 0 )
	{
	    Teuchos::broadcast<int,int>( 
		*d_block_comm, 0, Teuchos::Ptr<int>(&do_iterations) );
	}

	// Print iteration data.
	if ( d_global_comm->getRank() == 0 && d_num_iters % print_freq == 0 )
	{
	    std::cout << "SequentialMC Iteration " << d_num_iters 
		      << ": Residual = " 
		      << residual_norm/source_norm << std::endl;
	}

	// Barrier before proceeding.
	d_global_comm->barrier();
    }

    // Finalize.
    if ( d_primary_set )
    {
        // Recover the original solution if right preconditioned.
        if ( d_problem->isRightPrec() )
        {
            d_problem->applyRightPrec( *d_problem->getLHS(), 
                                       *d_problem->getLHS() );
        }

        // Check for convergence.
	if ( VT::normInf(*d_problem->getPrecResidual()) <= convergence_criteria )
	{
	    d_converged_status = 1;
	}
    }
    d_global_comm->barrier();

    // Broadcast convergence status to the blocks.
    Teuchos::broadcast<int,int>( 
	*d_block_comm, 0, Teuchos::Ptr<int>(&d_converged_status) );

    return Teuchos::as<bool>(d_converged_status);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the residual Monte Carlo problem.
 */
template<class Vector, class Matrix>
void SequentialMCSolverManager<Vector,Matrix>::buildResidualMonteCarloProblem()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_global_comm) );
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );

    // Generate the residual Monte Carlo problem on the primary set. The
    // unpreconditioned residual is the source.
    if ( d_primary_set )
    {
	Teuchos::RCP<Vector> delta_x = VT::clone( *d_problem->getLHS() );
	d_residual_problem = Teuchos::rcp(
	    new LinearProblemType( d_problem->getCompositeOperator(),
				   delta_x,
				   d_problem->getPrecResidual() ) );
    }
    d_global_comm->barrier();

    // Create the Monte Carlo direct solver for the residual problem.
    bool use_adjoint = true;
    if ( d_plist->isParameter("MC Type") )
    {
        if ( d_plist->get<std::string>("MC Type") == "Adjoint" )
        {
            use_adjoint = true;
        }
    }

    if ( use_adjoint )
    {
	d_mc_solver = Teuchos::rcp( 
	    new AdjointSolverManager<Vector,Matrix>(
		d_residual_problem, d_global_comm, d_plist) );

	// Get the block constant communicator.
	MCLS_CHECK( Teuchos::nonnull(d_mc_solver) );
	d_block_comm = 
	    Teuchos::rcp_dynamic_cast<AdjointSolverManager<Vector,Matrix> >(
		d_mc_solver)->blockComm();
    }

    MCLS_ENSURE( Teuchos::nonnull(d_mc_solver) );
    MCLS_ENSURE( Teuchos::nonnull(d_block_comm) );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_SEQUENTIALMCSOLVERMANAGER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_SequentialMCSolverManager_impl.hpp
//---------------------------------------------------------------------------//

