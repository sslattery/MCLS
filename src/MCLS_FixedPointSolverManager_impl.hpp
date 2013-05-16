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
 * \file MCLS_FixedPointSolverManager_impl.hpp
 * \author Stuart R. Slattery
 * \brief Fixed Point solver manager implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_FIXEDPOINTSOLVERMANAGER_IMPL_HPP
#define MCLS_FIXEDPOINTSOLVERMANAGER_IMPL_HPP

#include <string>

#include "MCLS_DBC.hpp"
#include "MCLS_FixedPointIterationFactory.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_Ptr.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Comm constructor. setProblem() must be called before solve().
 */
template<class Vector, class Matrix>
FixedPointSolverManager<Vector,Matrix>::FixedPointSolverManager( 
    const Teuchos::RCP<const Comm>& global_comm,
    const Teuchos::RCP<Teuchos::ParameterList>& plist )
    : d_global_comm( global_comm )
    , d_plist( plist )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_global_comm) );
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector, class Matrix>
FixedPointSolverManager<Vector,Matrix>::FixedPointSolverManager( 
    const Teuchos::RCP<LinearProblemType>& problem,
    const Teuchos::RCP<const Comm>& global_comm,
    const Teuchos::RCP<Teuchos::ParameterList>& plist )
    : d_problem( problem )
    , d_global_comm( global_comm )
    , d_plist( plist )
    , d_num_iters( 0 )
    , d_converged_status( 0 )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_problem) );
    MCLS_REQUIRE( Teuchos::nonnull(d_global_comm) );
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );

    // Create the fixed point iteration. Default to Richardson.
    std::string iteration_name = "Richardson";
    if ( d_plist->isParameter("Fixed Point Type") )
    {
	iteration_name = d_plist->get<std::string>("Fixed Point Type");
    }
    FixedPointIterationFactory<Vector,Matrix> fp_factory;
    d_fixed_point = 
        fp_factory.create( iteration_name, d_plist );
    d_fixed_point->setProblem( d_problem );   
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the valid parameters for this manager.
 */
template<class Vector, class Matrix>
Teuchos::RCP<const Teuchos::ParameterList> 
FixedPointSolverManager<Vector,Matrix>::getValidParameters() const
{
    // Create a parameter list with the fixed point parameters as a starting
    // point. 
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    if ( Teuchos::nonnull(d_fixed_point) )
    {
	plist->setParameters( *d_fixed_point->getValidParameters() );
    }

    plist->set<double>("Convergence Tolerance", 1.0);
    plist->set<int>("Maximum Iterations", 1000);
    plist->set<int>("Iteration Print Frequency", 10);
    plist->set<int>("Iteration Check Frequency", 1);
    plist->set<std::string>("Fixed Point Type", "Richardson");

    return plist;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the tolerance achieved on the last linear solve. This may be
 * less or more than the set convergence tolerance. 
 */
template<class Vector, class Matrix>
typename Teuchos::ScalarTraits<
    typename FixedPointSolverManager<Vector,Matrix>::Scalar>::magnitudeType 
FixedPointSolverManager<Vector,Matrix>::achievedTol() const
{
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType residual_norm = 
	Teuchos::ScalarTraits<Scalar>::zero();
    residual_norm = VT::normInf( *d_problem->getPrecResidual() );

    // Compute the source norm preconditioned if necessary.
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType source_norm = 0;
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

    return residual_norm;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the linear problem with the manager.
 */
template<class Vector, class Matrix>
void FixedPointSolverManager<Vector,Matrix>::setProblem( 
    const Teuchos::RCP<LinearProblem<Vector,Matrix> >& problem )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_global_comm) );
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );

    // Set the problem.
    d_problem = problem;

    // Create the fixed point iteration. Default to Richardson.
    std::string iteration_name = "Richardson";
    if ( d_plist->isParameter("Fixed Point Type") )
    {
	iteration_name = d_plist->get<std::string>("Fixed Point Type");
    }
    FixedPointIterationFactory<Vector,Matrix> fp_factory;
    d_fixed_point = 
        fp_factory.create( iteration_name, d_plist );

    d_fixed_point->setProblem( d_problem );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the parameters for the manager. The manager will modify this
 * list with default parameters that are not defined.
 */
template<class Vector, class Matrix>
void FixedPointSolverManager<Vector,Matrix>::setParameters( 
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
template<class Vector, class Matrix>
bool FixedPointSolverManager<Vector,Matrix>::solve()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_global_comm) );
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );

    // Get the convergence parameters on the primary set.
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType 
	convergence_criteria = 0;
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType source_norm = 0.0;

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
    d_converged_status = 0;

    // Iteration setup.
    d_fixed_point->setParameters( d_plist );
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

    // Set the residual.
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType residual_norm = 0;
    d_problem->updatePrecResidual();
    residual_norm = VT::normInf( *d_problem->getPrecResidual() );

    // Iterate.
    int do_iterations = 1;
    while( do_iterations )
    {
	// Update the iteration count.
	++d_num_iters;

        // Do a fixed point iteration.
        d_fixed_point->doOneIteration();

        // Check if we're done iterating.
        residual_norm = VT::normInf( *d_problem->getPrecResidual() );
        if ( d_num_iters % check_freq == 0 )
        {
            do_iterations = (residual_norm > convergence_criteria) &&
                            (d_num_iters < max_num_iters);
        }

	// Print iteration data.
	if ( d_global_comm->getRank() == 0 && d_num_iters % print_freq == 0 )
	{
	    std::cout << d_fixed_point->name() << " Iteration " << d_num_iters
		      << ": Residual = " 
		      << residual_norm/source_norm << std::endl;
	}

	// Barrier before proceeding.
	d_global_comm->barrier();
    }

    // Recover the original solution if right preconditioned.
    if ( d_problem->isRightPrec() )
    {
            Teuchos::RCP<Vector> temp = VT::clone(*d_problem->getLHS());
	    d_problem->applyRightPrec( *d_problem->getLHS(),
                                       *temp );
            VT::update( *d_problem->getLHS(),
                        Teuchos::ScalarTraits<Scalar>::zero(),
                        *temp,
                        Teuchos::ScalarTraits<Scalar>::one() );
    }

    // Check for convergence.
    if ( VT::normInf(*d_problem->getPrecResidual()) <= convergence_criteria )
    {
        d_converged_status = 1;
    }

    return Teuchos::as<bool>(d_converged_status);
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_FIXEDPOINTSOLVERMANAGER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_FixedPointSolverManager_impl.hpp
//---------------------------------------------------------------------------//

