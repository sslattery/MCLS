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
 * \file MCLS_AndersonSolverManager_impl.hpp
 * \author Stuart R. Slattery
 * \brief Anderson Acceleration solver manager implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ANDERSONSOLVERMANAGER_IMPL_HPP
#define MCLS_ANDERSONSOLVERMANAGER_IMPL_HPP

#include <string>
#include <iostream>
#include <iomanip>

#include "MCLS_DBC.hpp"
#include "MCLS_ThyraVectorExtraction.hpp"
#include "MCLS_MCSAStatusTest.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_Ptr.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_Time.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Comm constructor. setProblem() must be called before solve().
 */
template<class Vector, class Matrix, class RNG>
AndersonSolverManager<Vector,Matrix,RNG>::AndersonSolverManager( 
    const Teuchos::RCP<const Comm>& global_comm,
    const Teuchos::RCP<Teuchos::ParameterList>& plist )
    : d_global_comm( global_comm )
    , d_plist( plist )
    , d_model_evaluator( global_comm, plist )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_global_comm) );
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );
    d_plist->set( "Nonlinear Solver", "Anderson Accelerated Fixed-Point" );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector, class Matrix, class RNG>
AndersonSolverManager<Vector,Matrix,RNG>::AndersonSolverManager( 
    const Teuchos::RCP<LinearProblemType>& problem,
    const Teuchos::RCP<const Comm>& global_comm,
    const Teuchos::RCP<Teuchos::ParameterList>& plist )
    : d_problem( problem )
    , d_model_evaluator( global_comm,
			 plist,
			 d_problem->getOperator(), 
			 d_problem->getRHS(),
			 d_problem->getLeftPrec() )
    , d_global_comm( global_comm )
    , d_plist( plist )
    , d_nox_solver( new ::Thyra::NOXNonlinearSolver )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_global_comm) );
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );
    d_plist->set( "Nonlinear Solver", "Anderson Accelerated Fixed-Point" );
    d_nox_solver->setParameterList( d_plist );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the valid parameters for this manager.
 */
template<class Vector, class Matrix, class RNG>
Teuchos::RCP<const Teuchos::ParameterList> 
AndersonSolverManager<Vector,Matrix,RNG>::getValidParameters() const
{
    // Create a parameter list with the Monte Carlo solver parameters as a
    // starting point.
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    return plist;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the tolerance achieved on the last linear solve. This may be
 * less or more than the set convergence tolerance. 
 */
template<class Vector, class Matrix, class RNG>
typename Teuchos::ScalarTraits<
    typename AndersonSolverManager<Vector,Matrix,RNG>::Scalar>::magnitudeType 
AndersonSolverManager<Vector,Matrix,RNG>::achievedTol() const
{
    return 0.0;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the linear problem with the manager.
 */
template<class Vector, class Matrix, class RNG>
void AndersonSolverManager<Vector,Matrix,RNG>::setProblem( 
    const Teuchos::RCP<LinearProblemType>& problem )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_global_comm) );
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );

    d_problem = problem;
    d_model_evaluator->setProblem( d_prolem->getOperator(),
				   d_problem->getRHS(),
				   d_problem->getLeftPrec() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the parameters for the manager. The manager will modify this
 * list with default parameters that are not defined.
 */
template<class Vector, class Matrix, class RNG>
void AndersonSolverManager<Vector,Matrix,RNG>::setParameters( 
    const Teuchos::RCP<Teuchos::ParameterList>& params )
{
    MCLS_REQUIRE( Teuchos::nonnull(params) );

    // Set the parameters.
    d_plist = params;
    d_model_evaluator->setParameters( params );
    d_plist->set( "Nonlinear Solver", "Anderson Accelerated Fixed-Point" );
    d_nox_solver->setParameterList( params );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Solve the linear problem. Return true if the solution
 * converged. False if it did not.
 */
template<class Vector, class Matrix, class RNG>
bool AndersonSolverManager<Vector,Matrix,RNG>::solve()
{
    // Set the MCSA model evaluator with NOX.
    d_nox_solver->setModel(d_model_evaluator);

    // Create a Thyra vector from our initial guess.
    Teuchos::RCP< ::Thyra::VectorBase<double> > x0 = 
	ThyraVectorExtraction<Vector>::createThyraVector( d_problem->getLHS() );
    NOX::Thyra::Vector nox_x0( x0 );

    // Create the solve criteria.
    typename Teuchos::ScalarTraits<double>::magnitudeType tolerance = 1.0e-8;
    if ( d_plist->isParameter("Convergence Tolerance") )
    {
	tolerance = d_plist->get<double>("Convergence Tolerance");
    }
    Teuchos::RCP<MCSAStatusTest<Vector,Matrix> > tol_test= Teuchos::rcp(
	new MCSAStatusTest<Vector,Matrix>(tolerance) );
    int max_num_iters = 1000;
    if ( d_plist->isParameter("Maximum Iterations") )
    {
        max_num_iters = d_plist->get<int>("Maximum Iterations");
    }
    Teuchos::RCP<NOX::StatusTest::MaxIters> max_iter_test =
	Teuchos::rcp( new NOX::StatusTest::MaxIters(max_num_iters) );
    Teuchos::RCP<NOX::StatusTest::FiniteValue> finite_test =
	Teuchos::rcp(new NOX::StatusTest::FiniteValue);
    Teuchos::RCP<NOX::StatusTest::Combo> status_test =
	Teuchos::rcp( new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR) );
    status_test->add( tol_test );
    status_test->add( max_iter_test );
    status_test->add( finite_test );

    // Create the NOX group.
    Teuchos::RCP<NOX::Abstract::Group> nox_group = Teuchos::rcp(
	new NOX::Thyra::Group(nox_X0, d_model_evaluator) );

    // Create the NOX solver.
    NOX::Solver::Factory nox_factory;
    Teuchos::RCP<NOX::Solver::Generic> nox_solver = 
	nox_factory.buildSolver( nox_group, status_test, d_plist );

    // Solve the problem.
    NOX::StatusTest::StatusType solve_status = solver->solve();

    // Extract the solution.
    Teuchos::RCP< ::Thyra::NOXNonlinearSolver> thyra_nox_solver =
	Teuchos::rcp_dynamic_cast< ::Thyra::NOXNonlinearSolver>(solver);

    Teuchos::RCP<const NOX::Abstract::Vector> x = 
	thyra_nox_solver->getNOXSolver()->getSolutionGroup().getXPtr();

    Teuchos::RCP<const NOX::Thyra::Vector> nox_thyra_x =
	Teuchos::rcp_dynamic_cast<const NOX::Thyra::Vector>(x,true);

    d_problem->setLHS(
	ThyraVectorExtraction<Vector>::getVector( 
	    nox_thyra_x->getThyraRCPVector(), *d_problem->getLHS() );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_ANDERSONSOLVERMANAGER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_AndersonSolverManager_impl.hpp
//---------------------------------------------------------------------------//

