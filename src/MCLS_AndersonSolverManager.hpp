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
 * \file MCLS_AndersonSolverManager.hpp
 * \author Stuart R. Slattery
 * \brief Anderson Acceleration solver manager declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ANDERSONSOLVERMANAGER_HPP
#define MCLS_ANDERSONSOLVERMANAGER_HPP

#include <random>

#include "MCLS_config.hpp"
#include "MCLS_MCSAModelEvaluator.hpp"
#include "MCLS_LinearProblem.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_Xorshift.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_as.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_Time.hpp>

#include <NOX.H>
#include <NOX_Abstract_PrePostOperator.H>
#include <NOX_Solver_Generic.H>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class AndersonSolverManager
 * \brief Solver manager for Monte Carlo synthetic acceleration.
 */
template<class Vector, class Matrix, class RNG = Xorshift<> >
class AndersonSolverManager : public SolverManager<Vector,Matrix>
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                         vector_type;
    typedef Matrix                                         matrix_type;
    typedef SolverManager<vector_type,matrix_type>         Base;
    typedef VectorTraits<vector_type>                      VT;
    typedef typename VT::scalar_type                       Scalar;
    typedef LinearProblem<vector_type,matrix_type>         LinearProblemType;
    typedef Teuchos::Comm<int>                             Comm;
    //@}

    // Comm constructor. setProblem() must be called before solve().
    AndersonSolverManager( const Teuchos::RCP<const Comm>& global_comm,
			   const Teuchos::RCP<Teuchos::ParameterList>& plist );

    // Constructor.
    AndersonSolverManager( const Teuchos::RCP<LinearProblemType>& problem,
			   const Teuchos::RCP<const Comm>& global_comm,
			   const Teuchos::RCP<Teuchos::ParameterList>& plist );

    //! Destructor.
    ~AndersonSolverManager() { /* ... */ }

    //! Get the linear problem being solved by the manager.
    const LinearProblem<vector_type,matrix_type>& getProblem() const
    { return *d_problem; }

    // Get the valid parameters for this manager.
    Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;

    //! Get the current parameters being used for this manager.
    Teuchos::RCP<const Teuchos::ParameterList> getCurrentParameters() const
    { return d_plist; }

    // Get the tolerance achieved on the last linear solve. This may be less
    // or more than the set convergence tolerance.
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType achievedTol() const;

    // Get the number of iterations from the last linear solve.
    int getNumIters() const { return d_nox_solver->getNumIterations(); };

    // Set the linear problem with the manager.
    void setProblem( const Teuchos::RCP<LinearProblemType >& problem );

    // Set the parameters for the manager. The manager will modify this list
    // with default parameters that are not defined.
    void setParameters( const Teuchos::RCP<Teuchos::ParameterList>& params );

    // Solve the linear problem. Return true if the solution converged. False
    // if it did not.
    bool solve();

    //! Return if the last linear solve converged. 
    bool getConvergedStatus() const 
    { return d_nox_solver->getStatus(); }

  private:

    // Create the nox solver.
    void createNonlinearSolver();

  private:

    // Linear problem
    Teuchos::RCP<LinearProblemType> d_problem;

    // Global communicator.
    Teuchos::RCP<const Comm> d_global_comm;

    // Parameters.
    Teuchos::RCP<Teuchos::ParameterList> d_plist;

    // MCSA model evaluator.
    Teuchos::RCP<MCSAModelEvaluator<Vector,Matrix,RNG> > d_model_evaluator;

    // NOX solver.
    Teuchos::RCP<NOX::Solver::Generic> d_nox_solver;

#if HAVE_MCLS_TIMERS
    // Total solve timer.
    Teuchos::RCP<Teuchos::Time> d_solve_timer;
#endif
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_AndersonSolverManager_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_ANDERSONSOLVERMANAGER_HPP

//---------------------------------------------------------------------------//
// end MCLS_AndersonSolverManager.hpp
//---------------------------------------------------------------------------//

