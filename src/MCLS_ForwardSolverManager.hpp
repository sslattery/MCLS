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
 * \file MCLS_ForwardSolverManager.hpp
 * \author Stuart R. Slattery
 * \brief Forward Monte Carlo solver manager declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_FORWARDSOLVERMANAGER_HPP
#define MCLS_FORWARDSOLVERMANAGER_HPP

#include <random>

#include "MCLS_SolverManager.hpp"
#include "MCLS_LinearProblem.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_MatrixTraits.hpp"
#include "MCLS_MSODManager.hpp"
#include "MCLS_MCSolver.hpp"
#include "MCLS_UniformForwardSource.hpp"
#include "MCLS_ForwardDomain.hpp"
#include "MCLS_TallyTraits.hpp"
#include "MCLS_Xorshift.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_Comm.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class ForwardSolverManager
 * \brief Solver manager for analog forward Monte Carlo.
 */
template<class Vector, class Matrix, class RNG = Xorshift<> >
class ForwardSolverManager : public SolverManager<Vector,Matrix>
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
    typedef ForwardDomain<Vector,Matrix,RNG>        DomainType;
    typedef typename DomainType::TallyType          TallyType;
    typedef TallyTraits<TallyType>                  TT;
    typedef UniformForwardSource<DomainType>        SourceType;
    typedef RNG                                     rng_type;
    typedef Teuchos::Comm<int>                      Comm;
    //@}

    // Comm constructor. setProblem() must be called before solve().
    ForwardSolverManager( const Teuchos::RCP<const Comm>& global_comm,
			  const Teuchos::RCP<Teuchos::ParameterList>& plist,
                          bool internal_solver = false );

    // Constructor.
    ForwardSolverManager( const Teuchos::RCP<LinearProblemType>& problem,
			  const Teuchos::RCP<const Comm>& global_comm,
			  const Teuchos::RCP<Teuchos::ParameterList>& plist,
                          bool internal_solver = false );

    //! Destructor.
    ~ForwardSolverManager() { /* ... */ }

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

    //! Return if the last linear solve converged. The forward Monte Carlo
    //! solver is a direct solver, and therefore always converges in the
    //! iterative sense.
    bool getConvergedStatus() const
    { return true; }

    //! Get the block-constant communicator for this set.
    Teuchos::RCP<const Comm> blockComm() const { return d_msod_manager->blockComm(); }

  private:

    // Build the Monte Carlo domain from the provided linear problem.
    void buildMonteCarloDomain();

    // Build the Monte Carlo source from the provided linear problem.
    void buildMonteCarloSource();

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

    // MSOD Manager.
    Teuchos::RCP<MSODManager<SourceType> > d_msod_manager;

    // Monte Carlo set solver.
    Teuchos::RCP<MCSolver<SourceType> > d_mc_solver;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_ForwardSolverManager_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_FORWARDSOLVERMANAGER_HPP

//---------------------------------------------------------------------------//
// end MCLS_ForwardSolverManager.hpp
//---------------------------------------------------------------------------//

