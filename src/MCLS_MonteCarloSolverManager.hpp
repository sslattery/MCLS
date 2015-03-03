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
 * \file MCLS_MonteCarloSolverManager.hpp
 * \author Stuart R. Slattery
 * \brief Monte Carlo solver manager declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MONTECARLOSOLVERMANAGER_HPP
#define MCLS_MONTECARLOSOLVERMANAGER_HPP

#include <random>

#include "MCLS_config.hpp"
#include "MCLS_SolverManager.hpp"
#include "MCLS_LinearProblem.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_MatrixTraits.hpp"
#include "MCLS_MCSolver.hpp"
#include "MCLS_UniformAdjointSource.hpp"
#include "MCLS_AdjointTally.hpp"
#include "MCLS_UniformForwardSource.hpp"
#include "MCLS_ForwardTally.hpp"
#include "MCLS_AlmostOptimalDomain.hpp"
#include "MCLS_Xorshift.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_Time.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
// Monte Carlo Algorithm Tag Traits.
//---------------------------------------------------------------------------//

// Dispatch tags.
class ForwardTag {};
class AdjointTag {};

// Algorithm tag traits.
template<class Vector, class Matrix, class RNG, class Tag>
class MonteCarloTagTraits {};

// Forward specialization.
template<class Vector, class Matrix, class RNG>
class MonteCarloTagTraits<Vector,Matrix,RNG,ForwardTag>
{
  public:
    typedef ForwardTally<Vector>                              tally_type;
    typedef AlmostOptimalDomain<Vector,Matrix,RNG,tally_type> domain_type;
    typedef UniformForwardSource<domain_type>                 source_type;
};

// Adjoint specialization.
template<class Vector, class Matrix, class RNG>
class MonteCarloTagTraits<Vector,Matrix,RNG,AdjointTag>
{
  public:
    typedef AdjointTally<Vector>                              tally_type;
    typedef AlmostOptimalDomain<Vector,Matrix,RNG,tally_type> domain_type;
    typedef UniformAdjointSource<domain_type>                 source_type;
};

//---------------------------------------------------------------------------//
/*!
 * \class MonteCarloSolverManager
 * \brief Solver manager for analog Monte Carlo.
 */
template<class Vector,
	 class Matrix,
	 class MonteCarloTag = AdjointTag,
	 class RNG = Xorshift<> >
class MonteCarloSolverManager : public SolverManager<Vector,Matrix>
{
  public:

    //@{
    //! Typedefs.
    typedef SolverManager<Vector,Matrix>               Base;
    typedef Vector                                     vector_type;
    typedef VectorTraits<Vector>                       VT;
    typedef typename VT::scalar_type                   Scalar;
    typedef Matrix                                     matrix_type;
    typedef MatrixTraits<Vector,Matrix>                MT;
    typedef LinearProblem<Vector,Matrix>               LinearProblemType;
    typedef MonteCarloTag                              Tag;
    typedef MonteCarloTagTraits<Vector,Matrix,RNG,Tag> MCTT;
    typedef typename MCTT::domain_type                 DomainType;
    typedef typename MCTT::source_type                 SourceType;
    typedef RNG                                        rng_type;
    //@}

    // Parameter constructor. setProblem() must be called before solve().
    MonteCarloSolverManager( const Teuchos::RCP<Teuchos::ParameterList>& plist,
			     bool internal_solver = false );

    // Constructor.
    MonteCarloSolverManager( const Teuchos::RCP<LinearProblemType>& problem,
			     const Teuchos::RCP<Teuchos::ParameterList>& plist,
			     bool internal_solver = false );

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

    //! Return if the last linear solve converged. The Monte Carlo solver is a
    //! direct solver, and therefore always converges in the iterative sense.
    bool getConvergedStatus() const
    { return true; }

  private:

    // Get the composite operator for a given algorithm tag.
    Teuchos::RCP<const Matrix>
    getCompositeOperator( const double threshold,
			  ForwardTag ) const;
    Teuchos::RCP<const Matrix>
    getCompositeOperator( const double threshold,
			  AdjointTag ) const;

    // Initialize the tally for a solve for a given algorithm tag.
    void initializeTally( ForwardTag );
    void initializeTally( AdjointTag );

    // Build the Monte Carlo domain from the provided linear problem.
    void buildMonteCarloDomain();

    // Build the Monte Carlo source from the provided linear problem.
    void buildMonteCarloSource();

  private:

    // Linear problem
    Teuchos::RCP<LinearProblemType> d_problem;

    // Parameters.
    Teuchos::RCP<Teuchos::ParameterList> d_plist;

    // Boolean for internal solver (i.e. inside of MCSA).
    bool d_internal_solver;

    // Local domain for this proc.
    Teuchos::RCP<DomainType> d_domain;

    // Local source for this proc.
    Teuchos::RCP<SourceType> d_source;

    // Monte Carlo set solver.
    Teuchos::RCP<MCSolver<SourceType> > d_mc_solver;

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

#include "MCLS_MonteCarloSolverManager_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_MONTECARLOSOLVERMANAGER_HPP

//---------------------------------------------------------------------------//
// end MCLS_MonteCarloSolverManager.hpp
//---------------------------------------------------------------------------//

