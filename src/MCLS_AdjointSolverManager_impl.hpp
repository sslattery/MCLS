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
 * \file MCLS_AdjointSolverManager_impl.hpp
 * \author Stuart R. Slattery
 * \brief Adjoint Monte Carlo solver manager implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ADJOINTSOLVERMANAGER_IMPL_HPP
#define MCLS_ADJOINTSOLVERMANAGER_IMPL_HPP

#include "MCLS_DBC.hpp"

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Comm constructor. setProblem() and setParameters() must be called
 * before solve(). 
 */
template<class Vector, class Matrix>
AdjointSolverManager<Vector,Matrix>::AdjointSolverManager( 
    const Teuchos::RCP<const Comm>& global_comm,
    const Teuchos::RCP<Teuchos::ParameterList>& plist )
    : d_global_comm( global_comm )
    , d_plist( plist )
{
    Require( !d_global_comm.is_null() );
    Require( !d_plist.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector, class Matrix>
AdjointSolverManager<Vector,Matrix>::AdjointSolverManager( 
    const Teuchos::RCP<LinearProblemType>& problem,
    const Teuchos::RCP<const Comm>& global_comm,
    const Teuchos::RCP<Teuchos::ParameterList>& plist )
    : d_problem( problem )
    , d_global_comm( global_comm )
    , d_plist( plist )
    , d_primary_set( !d_problem.is_null() )
{
    Require( !d_global_comm.is_null() );
    Require( !d_plist.is_null() );

    buildMonteCarloDomain();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the valid parameters for this manager.
 */
template<class Vector, class Matrix>
Teuchos::RCP<const Teuchos::ParameterList> 
AdjointSolverManager<Vector,Matrix>::getValidParameters() const
{
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();

    // Set the list values to the default code values. Put zero if no default.
    plist->set<double>("Weight Cutoff", 0.0);
    plist->set<int>("MC Check Frequency", 1);
    plist->set<int>("MC Buffer Size", 1000);
    plist->set<bool>("Reproducible MC Mode", false);
    plist->set<int>("Overlap Size",0);
    plist->set<int>("Random Number Seed", 433494437);
    plist->set<int>("Number of Sets", 0);

    return plist;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the tolerance achieved on the last linear solve. This may be
 * less or more than the set convergence tolerance.
 */
template<class Vector, class Matrix>
typename Teuchos::ScalarTraits<
    typename AdjointSolverManager<Vector,Matrix>::Scalar>::magnitudeType 
AdjointSolverManager<Vector,Matrix>::achievedTol() const
{
    // Here we'll simply return the source weighted norm of the residual after
    // solution. This will give us a measure of the stochastic error generated
    // by the Monte Carlo solve. 
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType residual_norm = 
	Teuchos::ScalarTraits<Scalar>::zero();

    // We only do this on the primary set where the linear problem exists.
    if ( d_primary_set )
    {
	d_problem->updateResidual();
	residual_norm = VT::normInf( *d_problem->getResidual() );
	residual_norm /= VT::normInf( *d_problem->getRHS() );
    }
    d_global_comm->barrier();

    return residual_norm;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the number of iterations from the last linear solve. This is a
 * direct solver and therefore does not do any iterations.
 */
template<class Vector, class Matrix>
int AdjointSolverManager<Vector,Matrix>::getNumIters() const
{
    return 0;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the linear problem with the manager.
 */
template<class Vector, class Matrix>
void AdjointSolverManager<Vector,Matrix>::setProblem( 
    const Teuchos::RCP<LinearProblem<Vector,Matrix> >& problem )
{
    Require( !d_global_comm.is_null() );
    Require( !d_plist.is_null() );

    d_problem = problem;
    d_primary_set = !d_problem.is_null();

    // Calling this method implies that a new linear operator has been
    // provided. Update the parallel domain with this operator.
    buildMonteCarloDomain();
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the parameters for the manager. The manager will modify this
    list with default parameters that are not defined. 
*/
template<class Vector, class Matrix>
void AdjointSolverManager<Vector,Matrix>::setParameters( 
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
bool AdjointSolverManager<Vector,Matrix>::solve()
{    
    Require( !d_global_comm.is_null() );
    Require( !d_plist.is_null() );
    Require( !d_msod_manager.is_null() );
    Require( !d_mc_solver.is_null() );

    // Get the domain tally.
    Teuchos::RCP<TallyType> tally = 
	d_msod_manager->localDomain()->domainTally();
    Check( !tally.is_null() );

    // Set the primary set's tally vector to the LHS of the linear problem. We
    // need to do this because the MSODManager just creates some arbitrary
    // tally vector on all sets. In the primary set, we want to tally directly
    // into the linear problem vector to get the solution. The MSODManager
    // builds the tally vectors from the map of the solution vector, so the
    // set export and block reduction parallel operations are still valid.
    if ( d_primary_set )
    {
	TT::setBaseVector( *tally, d_problem->getLHS() );
    }
    d_global_comm->barrier();

    // Set the local domain with the solver.
    d_mc_solver->setDomain( d_msod_manager->localDomain() );

    // Build the global source. We assume the RHS of the linear system changes
    // with each solve.
    buildMonteCarloSource();

    // Set the local source with the solver.
    d_mc_solver->setSource( d_msod_manager->localSource() );

    // Solve the problem.
    d_mc_solver->solve();

    // Barrier before proceeding.
    d_global_comm->barrier();

    // Combine the tallies across the blocks.
    TT::combineBlockTallies( *tally, d_msod_manager->blockComm() );

    // Normalize the tallies by the number of sets.
    TT::normalize( *tally, d_msod_manager->numSets() );

    // Barrier before exiting.
    d_global_comm->barrier();

    // If we're right preconditioned then we have to recover the original
    // solution. 
    if ( d_problem->isRightPrec() )
    {
	d_problem->applyRightPrec( *d_problem->getLHS(), *d_problem->getLHS() );
    }

    // This is a direct solve and therefore always converged in the iterative
    // sense. 
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the Monte Carlo domain from the provided linear problem.
 */
template<class Vector, class Matrix>
void AdjointSolverManager<Vector,Matrix>::buildMonteCarloDomain()
{
    Require( !d_global_comm.is_null() );
    Require( !d_plist.is_null() );

    // Build the MSOD manager.
    d_msod_manager = Teuchos::rcp( 
	new MSODManager<SourceType>(d_primary_set, d_global_comm, *d_plist) );

    // Build the Monte Carlo set solver.
    if ( d_mc_solver.is_null() )
    {
	d_mc_solver = Teuchos::rcp(
	    new MCSolver<SourceType>(d_msod_manager->setComm(), d_plist) );
    }

    // Set a global scope variable for the primary domain.
    Teuchos::RCP<DomainType> primary_domain;

    // Build the primary domain in the primary set using the composite
    // operator.
    if ( d_primary_set )
    {
	primary_domain = Teuchos::rcp( 
	    new DomainType( d_problem->getCompositeOperator(),
			    d_problem->getLHS(),
			    *d_plist ) );
    }
    d_global_comm->barrier();

    // Build the global MSOD Monte Carlo domain from the primary set.
    d_msod_manager->setDomain( primary_domain );

    Ensure( !d_msod_manager.is_null() );
    Ensure( !d_msod_manager->localDomain().is_null() );
    Ensure( !d_mc_solver.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the Monte Carlo source from the provided linear problem.
 */
template<class Vector, class Matrix>
void AdjointSolverManager<Vector,Matrix>::buildMonteCarloSource()
{
    Require( !d_global_comm.is_null() );
    Require( !d_plist.is_null() );
    Require( !d_msod_manager.is_null() );
    Require( !d_msod_manager->localDomain().is_null() );
    Require( !d_mc_solver.is_null() );

    // Set a global scope variable for the primary source.
    Teuchos::RCP<SourceType> primary_source;

    // Build the primary source in the primary set.
    if ( d_primary_set )
    {
	// Left precondition the source if necessary.
	if ( d_problem->isLeftPrec() )
	{
	    Teuchos::RCP<Vector> prec_src = VT::clone( *d_problem->getRHS() );
	    d_problem->applyLeftPrec( *d_problem->getRHS(), *prec_src );

	    primary_source = Teuchos::rcp(
		new SourceType( 
		    prec_src,
		    d_msod_manager->localDomain(),
		    d_mc_solver->rngControl(),
		    d_msod_manager->setComm(),
		    d_global_comm->getSize(),
		    d_global_comm->getRank(),
		    *d_plist ) );
	}
	else
	{
	    primary_source = Teuchos::rcp(
		new SourceType( 
		    Teuchos::rcp_const_cast<Vector>(d_problem->getRHS()),
		    d_msod_manager->localDomain(),
		    d_mc_solver->rngControl(),
		    d_msod_manager->setComm(),
		    d_global_comm->getSize(),
		    d_global_comm->getRank(),
		    *d_plist ) );
	}
    }
    d_global_comm->barrier();

    // Build the global MSOD Monte Carlo source from the primary set.
    d_msod_manager->setSource( primary_source, d_mc_solver->rngControl() );

    Ensure( !d_msod_manager->localSource().is_null() );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_ADJOINTSOLVERMANAGER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_AdjointSolverManager_impl.hpp
//---------------------------------------------------------------------------//

