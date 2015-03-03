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
 * \file MCLS_MonteCarloSolverManager_impl.hpp
 * \author Stuart R. Slattery
 * \brief Carlo solver manager implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MONTECARLOSOLVERMANAGER_IMPL_HPP
#define MCLS_MONTECARLOSOLVERMANAGER_IMPL_HPP

#include "MCLS_DBC.hpp"

#include <Teuchos_TimeMonitor.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Parameter constructor. setProblem() and setParameters() must be called
 * before solve(). 
 */
template<class Vector, class Matrix, class AlgorithmTag, class RNG>
MonteCarloSolverManager<Vector,Matrix,AlgorithmTag,RNG>::MonteCarloSolverManager( 
    const Teuchos::RCP<Teuchos::ParameterList>& plist,
    bool internal_solver )
    : d_plist( plist )
    , d_internal_solver( internal_solver )
#if HAVE_MCLS_TIMERS
    , d_solve_timer( Teuchos::TimeMonitor::getNewCounter("MCLS: MC Solve") )
#endif
{
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector, class Matrix, class AlgorithmTag, class RNG>
MonteCarloSolverManager<Vector,Matrix,AlgorithmTag,RNG>::MonteCarloSolverManager( 
    const Teuchos::RCP<LinearProblemType>& problem,
    const Teuchos::RCP<Teuchos::ParameterList>& plist,
    bool internal_solver )
    : d_problem( problem )
    , d_plist( plist )
    , d_internal_solver( internal_solver )
#if HAVE_MCLS_TIMERS
    , d_solve_timer( Teuchos::TimeMonitor::getNewCounter("MCLS: MC Solve") )
#endif
{
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );

    buildMonteCarloDomain();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the valid parameters for this manager.
 */
template<class Vector, class Matrix, class AlgorithmTag, class RNG>
Teuchos::RCP<const Teuchos::ParameterList> 
MonteCarloSolverManager<Vector,Matrix,AlgorithmTag,RNG>::getValidParameters() const
{
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();

    // Set the list values to the default code values. Put zero if no default.
    plist->set<double>("History Length", 10);
    plist->set<int>("MC Check Frequency", 1000);
    plist->set<int>("MC Buffer Size", 1000);
    plist->set<double>("Neumann Relaxation", 1.0);
    return plist;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the tolerance achieved on the last linear solve. This may be
 * less or more than the set convergence tolerance.
 */
template<class Vector, class Matrix, class AlgorithmTag, class RNG>
typename Teuchos::ScalarTraits<
    typename MonteCarloSolverManager<Vector,Matrix,AlgorithmTag,RNG>::Scalar>::magnitudeType 
MonteCarloSolverManager<Vector,Matrix,AlgorithmTag,RNG>::achievedTol() const
{
    // Here we'll simply return the source weighted norm of the residual after
    // solution. This will give us a measure of the stochastic error generated
    // by the Monte Carlo solve. 
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType residual_norm = 
	Teuchos::ScalarTraits<Scalar>::zero();

    d_problem->updateResidual();
    residual_norm = VT::normInf( *d_problem->getResidual() );
    residual_norm /= VT::normInf( *d_problem->getRHS() );

    return residual_norm;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the number of iterations from the last linear solve. This is a
 * direct solver and therefore does not do any iterations.
 */
template<class Vector, class Matrix, class AlgorithmTag, class RNG>
int MonteCarloSolverManager<Vector,Matrix,AlgorithmTag,RNG>::getNumIters() const
{
    return 0;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the linear problem with the manager.
 */
template<class Vector, class Matrix, class AlgorithmTag, class RNG>
void MonteCarloSolverManager<Vector,Matrix,AlgorithmTag,RNG>::setProblem( 
    const Teuchos::RCP<LinearProblem<Vector,Matrix> >& problem )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );

    // Determine if the linear operator has changed. It is presumed the
    // preconditioners are bound to the linear operator and will therefore
    // change when the operator change. The mechanism here for determining if
    // the operator has changed is checking if the memory address is the
    // same. This may not be the best way to check.
    int update_operator = 1;
    if ( Teuchos::nonnull(d_problem) )
    {
	if ( d_problem->getOperator().getRawPtr() == 
	     problem->getOperator().getRawPtr() )
	{
	    update_operator = 0;
	}
    }

    // Set the problem.
    d_problem = problem;

    // Update the parallel domain with this operator if it has changed.
    if ( update_operator )
    {
        buildMonteCarloDomain();
    }
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the parameters for the manager. The manager will modify this
    list with default parameters that are not defined. 
*/
template<class Vector, class Matrix, class AlgorithmTag, class RNG>
void MonteCarloSolverManager<Vector,Matrix,AlgorithmTag,RNG>::setParameters( 
    const Teuchos::RCP<Teuchos::ParameterList>& params )
{
    MCLS_REQUIRE( Teuchos::nonnull(params) );
    d_plist = params;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Solve the linear problem. Return true if the solution
 * converged. False if it did not.
 */
template<class Vector, class Matrix, class AlgorithmTag, class RNG>
bool MonteCarloSolverManager<Vector,Matrix,AlgorithmTag,RNG>::solve()
{    
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );
    MCLS_REQUIRE( Teuchos::nonnull(d_mc_solver) );

#if HAVE_MCLS_TIMERS
    // Start the solve timer.
    Teuchos::TimeMonitor solve_monitor( *d_solve_timer );
#endif

    // Build the global source. We assume the RHS of the linear system changes
    // with each solve.
    buildMonteCarloSource();

    // Initialize the tally.
    initializeTally( AlgorithmTag() );
    
    // Solve the Monte Carlo problem over the set.
    d_mc_solver->solve();

    // If we're right preconditioned then we have to recover the original
    // solution.
    if ( d_problem->isRightPrec() && !d_internal_solver )
    {
	Teuchos::RCP<Vector> temp = VT::clone(*d_problem->getLHS());
	d_problem->applyRightPrec( *d_problem->getLHS(), *temp );
	VT::update( *d_problem->getLHS(),
		    Teuchos::ScalarTraits<Scalar>::zero(),
		    *temp,
		    Teuchos::ScalarTraits<Scalar>::one() );
    }

    // This is a direct solve and therefore always converged in the iterative
    // sense. 
    return true;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the composite operator for a given algorithm tag. Forward
 * overload.
 */
template<class Vector, class Matrix, class AlgorithmTag, class RNG>
Teuchos::RCP<const Matrix>
MonteCarloSolverManager<Vector,Matrix,AlgorithmTag,RNG>::getCompositeOperator(
    const double threshold, ForwardTag ) const
{
    return d_problem->getCompositeOperator(threshold);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the composite operator for a given algorithm tag. Adjoint
 * overload.
 */
template<class Vector, class Matrix, class AlgorithmTag, class RNG>
Teuchos::RCP<const Matrix>
MonteCarloSolverManager<Vector,Matrix,AlgorithmTag,RNG>::getCompositeOperator(
    const double threshold, AdjointTag ) const
{
    return d_problem->getTransposeCompositeOperator(threshold);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Initialize the tally for a solve. Forward overload.
 */
template<class Vector, class Matrix, class AlgorithmTag, class RNG>
void MonteCarloSolverManager<Vector,Matrix,AlgorithmTag,RNG>::initializeTally(
    ForwardTag )
{
    // Get a copy of the source so we can modify it.
    Teuchos::RCP<Vector> rhs_copy = VT::deepCopy( *d_problem->getRHS() );

    // Left precondition the source if necessary.
    if ( d_problem->isLeftPrec() && !d_internal_solver )
    {
	Teuchos::RCP<Vector> temp = VT::clone(*rhs_copy);
	d_problem->applyLeftPrec( *rhs_copy, *temp );
	VT::update( *rhs_copy,
		    Teuchos::ScalarTraits<Scalar>::zero(),
		    *temp,
		    Teuchos::ScalarTraits<Scalar>::one() );
    }

    // Scale by the Neumann relaxation parameter.
    if ( d_plist->isParameter("Neumann Relaxation") )
    {
	double omega = d_plist->get<double>("Neumann Relaxation");
	VT::scale( *rhs_copy, omega );
    }

    d_domain->domainTally()->setSource( rhs_copy );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Initialize the tally for a solve. Adjoint overload.
 */
template<class Vector, class Matrix, class AlgorithmTag, class RNG>
void MonteCarloSolverManager<Vector,Matrix,AlgorithmTag,RNG>::initializeTally(
    AdjointTag )
{ /* ... */ }

//---------------------------------------------------------------------------//
/*!
 * \brief Build the Monte Carlo domain from the provided linear problem.
 */
template<class Vector, class Matrix, class AlgorithmTag, class RNG>
void MonteCarloSolverManager<Vector,Matrix,AlgorithmTag,RNG>::buildMonteCarloDomain()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );

    // Build the Monte Carlo set solver.
    if ( Teuchos::is_null(d_mc_solver) )
    {
	d_mc_solver = Teuchos::rcp(
	    new MCSolver<SourceType>(
		MT::getComm(*d_problem->getOperator()),
		MT::getComm(*d_problem->getOperator())->getRank(),
		d_plist) );
    }

    // Build the primary domain in the primary set using the tranposed
    // composite operator.
    double threshold = 0.0;
    if ( d_plist->isParameter("Composite Operator Threshold") )
    {
	threshold =  d_plist->get<double>("Composite Operator Threshold");
    }
    d_domain = Teuchos::rcp( 
	new DomainType( getCompositeOperator(threshold,AlgorithmTag()),
			d_problem->getLHS(),
			*d_plist ) );

    // Set the local domain with the monte carlo solver.
    d_mc_solver->setDomain( d_domain );

    MCLS_ENSURE( Teuchos::nonnull(d_domain) );
    MCLS_ENSURE( Teuchos::nonnull(d_mc_solver) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the Monte Carlo source from the provided linear problem.
 */
template<class Vector, class Matrix, class AlgorithmTag, class RNG>
void MonteCarloSolverManager<Vector,Matrix,AlgorithmTag,RNG>::buildMonteCarloSource()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );
    MCLS_REQUIRE( Teuchos::nonnull(d_domain()) );
    MCLS_REQUIRE( Teuchos::nonnull(d_mc_solver) );

    // Get a copy of the source so we can modify it.
    Teuchos::RCP<Vector> rhs_copy = VT::deepCopy( *d_problem->getRHS() );

    // Left precondition the source if necessary.
    if ( d_problem->isLeftPrec() && !d_internal_solver )
    {
	Teuchos::RCP<Vector> temp = VT::clone(*rhs_copy);
	d_problem->applyLeftPrec( *rhs_copy, *temp );
	VT::update( *rhs_copy,
		    Teuchos::ScalarTraits<Scalar>::zero(),
		    *temp,
		    Teuchos::ScalarTraits<Scalar>::one() );
    }

    // Scale by the Neumann relaxation parameter.
    if ( d_plist->isParameter("Neumann Relaxation") )
    {
	double omega = d_plist->get<double>("Neumann Relaxation");
	VT::scale( *rhs_copy, omega );
    }

    // Build the source.
    d_source = Teuchos::rcp( new SourceType(rhs_copy,d_domain,*d_plist) );
    MCLS_ENSURE( Teuchos::nonnull(d_source) );

    // Set the local source with the solver.
    d_mc_solver->setSource( d_source );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_MONTECARLOSOLVERMANAGER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_MonteCarloSolverManager_impl.hpp
//---------------------------------------------------------------------------//

