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
 * \file MCLS_MultilevelSolverManager.hpp
 * \author Stuart R. Slattery
 * \brief Multilevel Monte Carlo solver manager declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MULTILEVELSOLVERMANAGER_HPP
#define MCLS_MULTILEVELSOLVERMANAGER_HPP

#include "MCLS_SolverManager.hpp"
#include "MCLS_LinearProblem.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_MatrixTraits.hpp"
#include "MCLS_AdjointSolverManager.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_Comm.hpp>

#include <MLAPI_MultiLevelSA.h>
#include <MLAPI_Operator.h>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class MultilevelSolverManager
 * \brief Solver manager for multilevel Monte Carlo.
 */
template<class Vector, class Matrix>
class MultilevelSolverManager : public SolverManager<Vector,Matrix>
{
  public:

    //@{
    //! Typedefs.
    typedef SolverManager<Vector,Matrix>            Base;
    typedef Vector                                  vector_type;
    typedef VectorTraits<Vector>                    VT;
    typedef typename VT::scalar_type                Scalar;
    typedef Matrix                                  matrix_type;
    typedef MatrixTraits<Vector,Matrix>             MT;
    typedef LinearProblem<Vector,Matrix>            LinearProblemType;
    typedef Teuchos::Comm<int>                      Comm;
    //@}

    // Comm constructor. setProblem() must be called before solve().
    MultilevelSolverManager( const Teuchos::RCP<const Comm>& global_comm,
			     const Teuchos::RCP<Teuchos::ParameterList>& plist,
			     bool internal_solver = false );

    // Constructor.
    MultilevelSolverManager( const Teuchos::RCP<LinearProblemType>& problem,
			     const Teuchos::RCP<const Comm>& global_comm,
			     const Teuchos::RCP<Teuchos::ParameterList>& plist,
			     bool internal_solver = false );

    //! Destructor.
    ~MultilevelSolverManager() { /* ... */ }

    //! Get the linear problem being solved by the manager.
    const LinearProblem<Vector,Matrix>& getProblem() const
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
    int getNumIters() const;

    // Set the linear problem with the manager.
    void setProblem( 
	const Teuchos::RCP<LinearProblem<Vector,Matrix> >& problem );

    // Set the parameters for the manager. The manager will modify this list
    // with default parameters that are not defined.
    void setParameters( const Teuchos::RCP<Teuchos::ParameterList>& params );

    // Solve the linear problem. Return true if the solution converged. False
    // if it did not.
    bool solve();

    //! Return if the last linear solve converged. The multilevel Monte Carlo
    //! solver is a direct solver, and therefore always converges in the
    //! iterative sense.
    bool getConvergedStatus() const
    { return true; }

    //! Get the block-constant communicator for this set.
    Teuchos::RCP<const Comm> blockComm() const { return d_mc_solver->blockComm(); }

  private:

    // Build the multigrid hierarchy.
    void buildOperatorHierarchy();

    // Build the residual Hierarchy.
    void buildResidualHierarchy();

  private:

    // Linear problem
    Teuchos::RCP<LinearProblemType> d_problem;

    // Global communicator.
    Teuchos::RCP<const Comm> d_global_comm;

    // Parameters.
    Teuchos::RCP<Teuchos::ParameterList> d_plist;

    // Boolean for internal solver (i.e. inside of MCSA).
    bool d_internal_solver;

    // Primary set indicator.
    bool d_primary_set;

    // Number of levels.
    int d_num_levels;

    // ML interface.
    Teuchos::RCP<MLAPI::MultiLevelSA> d_mlapi;

    // Operator hierarchy local inverse diagonal copies.
    Teuchos::Array<Teuchos::RCP<Vector> > d_diagonal_inv;

    // Diagonally scaled ML_Operators.
    Teuchos::Array<ML_Operator*> d_scaled_ops;

    // Diagonally scaled MLAPI operator hierarchy.
    Teuchos::Array<Teuchos::RCP<MLAPI::Operator> > d_A;

    // Residual hierarchy.
    Teuchos::Array<Teuchos::RCP<Vector> > d_r;

    // Adjoint Monte Carlo solver.
    Teuchos::RCP<AdjointSolverManager<Vector,Matrix> > d_mc_solver;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_MultilevelSolverManager_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_MULTILEVELSOLVERMANAGER_HPP

//---------------------------------------------------------------------------//
// end MCLS_MultilevelSolverManager.hpp
//---------------------------------------------------------------------------//

