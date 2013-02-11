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
 * \file MCLS_SolverManagerAdapter_impl.hpp
 * \author Stuart R. Slattery
 * \brief Linear solver manager adapter class for Thyra blocked systems.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SOLVERMANAGERADAPTER_IMPL_HPP
#define MCLS_SOLVERMANAGERADAPTER_IMPL_HPP

#include <algorithm>

#include <MCLS_DBC.hpp>

#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_ScalarTraits.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class MultiVector, class Matrix>
SolverManagerAdapter<MultiVector,Matrix>::SolverManagerAdapter( 
    const Teuchos::RCP<SolverManager<Vector,Matrix> >& solver )
    : d_solver( solver )
{
    Require( Teuchos::nonnull(d_solver) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the linear problem with the manager.
 */
template<class MultiVector, class Matrix>
void SolverManagerAdapter<MultiVector,Matrix>::setProblem( 
    const Teuchos::RCP<LinearProblemAdapter<MultiVector,Matrix> >& problem )
{
    Require( Teuchos::nonnull(problem) );
    d_problem = problem;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Solve the blocked linear problem. 
 */
template<class MultiVector, class Matrix>
Thyra::SolveStatus<typename SolverManagerAdapter<MultiVector,Matrix>::Scalar> 
SolverManagerAdapter<MultiVector,Matrix>::solve()
{
    Require( Teuchos::nonnull(d_problem) );

    Teuchos::Time timer("");
    bool converged = true;
    int num_iters = 0;
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType achieved_tol = 
	Teuchos::ScalarTraits<Scalar>::zero();
    int num_problems = d_problem->getNumSubProblems();
    Teuchos::RCP<LinearProblem<Vector,Matrix> > linear_problem;

    // Solve the individual linear problems.
    timer.start(true);
    for ( int n = 0; n < num_problems; ++n )
    {
	linear_problem = d_problem->getSubProblem( n );
	d_solver->setProblem( linear_problem );

	d_solver->solve();

	num_iters += d_solver->getNumIters();

	if ( !d_solver->getConvergedStatus() )
	{
	    converged = false;
	}

	achieved_tol = std::max( d_solver->achievedTol(), achieved_tol );
    }
    timer.stop();

    // Collect the solve results.
    Thyra::SolveStatus<typename VectorTraits<Vector>::scalar_type> solve_status;
    solve_status.solveStatus = ( converged ? 
				 Thyra::SOLVE_STATUS_CONVERGED :
				 Thyra::SOLVE_STATUS_UNCONVERGED );
    solve_status.achievedTol = achieved_tol;

    // Report the solve results.
    std::ostringstream ossmessage;
    ossmessage << "MCLS solver \""<< d_solver->description()
	       << "\" returned a solve status of \""
	       << toString(solve_status.solveStatus) << "\""
	       << " for " << num_problems << "RHSs using "
	       << num_iters << " cumulative iterations"
	       << " for an average of " << num_iters/num_problems 
	       << " iterations/RHS and with total CPU time of " 
	       << timer.totalElapsedTime() << " sec.";

    solve_status.message = ossmessage.str();
    
    // Add extra parameters from solve to status.
    if ( solve_status.extraParameters.is_null() ) 
    {
	solve_status.extraParameters = Teuchos::parameterList();
    }
    solve_status.extraParameters->set("MCLS/Iteration Count", num_iters);
    solve_status.extraParameters->set("Iteration Count", num_iters);
    solve_status.extraParameters->set("MCLS/Achieved Tolerance", 
				      solve_status.achievedTol);

    return solve_status;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_SOLVERMANAGERADAPTER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_SolverManagerAdapter_impl.hpp
//---------------------------------------------------------------------------//

