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
#include "MCLS_VectorExport.hpp"
#include "MCLS_Estimators.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_Ptr.hpp>

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
    MCLS_REQUIRE( !d_global_comm.is_null() );
    MCLS_REQUIRE( !d_plist.is_null() );
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
    MCLS_REQUIRE( !d_global_comm.is_null() );
    MCLS_REQUIRE( !d_plist.is_null() );

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
    plist->set<double>("Weight Cutoff", 1.0e-4);
    plist->set<int>("MC Check Frequency", 1000);
    plist->set<int>("MC Buffer Size", 1000);
    plist->set<bool>("Reproducible MC Mode", false);
    plist->set<int>("Overlap Size",0);
    plist->set<int>("Random Number Seed", 433494437);
    plist->set<int>("Number of Sets", 1);
    plist->set<double>("Neumann Relaxation", 1.0);
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
    MCLS_REQUIRE( !d_global_comm.is_null() );
    MCLS_REQUIRE( !d_plist.is_null() );

    d_primary_set = Teuchos::nonnull( problem );

    // Determine if the linear operator has changed. It is presumed the
    // preconditioners are bound to the linear operator and will therefore
    // change when the operator change. The mechanism here for determining if
    // the operator has changed is checking if the memory address is the
    // same. This may not be the best way to check.
    int update_operator = 1;
    if ( d_primary_set )
    {
        if ( Teuchos::nonnull(d_problem) )
        {
            if ( d_problem->getOperator().getRawPtr() == 
                 problem->getOperator().getRawPtr() )
            {
                update_operator = 0;
            }
        }
    }

    // Set the problem.
    d_problem = problem;

    // Get the block comm if it exists.
    Teuchos::RCP<const Comm> block_comm;
    if ( Teuchos::nonnull(d_msod_manager) )
    {
        block_comm = d_msod_manager->blockComm();
    }
    if ( Teuchos::nonnull(block_comm) )
    {
        Teuchos::broadcast<int,int>( 
            *block_comm, 0, Teuchos::Ptr<int>(&update_operator) );
    }
    else
    {
        update_operator = 1;
    }

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
template<class Vector, class Matrix>
void AdjointSolverManager<Vector,Matrix>::setParameters( 
    const Teuchos::RCP<Teuchos::ParameterList>& params )
{
    MCLS_REQUIRE( !params.is_null() );
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
    MCLS_REQUIRE( !d_global_comm.is_null() );
    MCLS_REQUIRE( !d_plist.is_null() );
    MCLS_REQUIRE( !d_msod_manager.is_null() );
    MCLS_REQUIRE( !d_mc_solver.is_null() );

    // Get the estimator type.
    int estimator = COLLISION;
    if ( d_plist->isParameter("Estimator Type") )
    {
        estimator = d_plist->get<int>("Estimator Type");
    }

    // Get the domain tally.
    Teuchos::RCP<TallyType> tally = 
	d_msod_manager->localDomain()->domainTally();
    MCLS_CHECK( Teuchos::nonnull(tally) );

    // Set the primary set's base vector to the LHS of the linear problem. 
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

    // Combine the tallies across the blocks and normalize the tallies by the
    // number of sets.
    TT::combineBlockTallies( *tally, 
                             d_msod_manager->blockComm(),
                             d_msod_manager->numSets() );

    // Finalize.
    if ( d_primary_set )
    {
        // If we used the expected value estimator we have to add the RHS into
        // the solution.
        if ( EXPECTED_VALUE == estimator )
        {
            if ( d_problem->isLeftPrec() )
            {
                Teuchos::RCP<Vector> temp = VT::clone( *d_problem->getRHS() );
                d_problem->applyLeftPrec( *d_problem->getRHS(), *temp );
                VT::update( *d_problem->getLHS(),
                            Teuchos::ScalarTraits<Scalar>::one(),
                            *temp,
                            Teuchos::ScalarTraits<Scalar>::one() );
            }
            else
            {
                VT::update( *d_problem->getLHS(),
                            Teuchos::ScalarTraits<Scalar>::one(),
                            *d_problem->getRHS(),
                            Teuchos::ScalarTraits<Scalar>::one() );
            }
        }

        // If we're right preconditioned then we have to recover the original
        // solution on the primary set.
	if ( d_problem->isRightPrec() )
	{
            Teuchos::RCP<Vector> temp = VT::clone(*d_problem->getLHS());
	    d_problem->applyRightPrec( *d_problem->getLHS(), *temp );
            VT::update( *d_problem->getLHS(),
                        Teuchos::ScalarTraits<Scalar>::zero(),
                        *temp,
                        Teuchos::ScalarTraits<Scalar>::one() );
	}

        // Export the LHS to the original decomposition.
        d_problem->exportLHS();
    }
    d_global_comm->barrier();


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
    MCLS_REQUIRE( !d_global_comm.is_null() );
    MCLS_REQUIRE( !d_plist.is_null() );

    // Build the MSOD manager.
    d_msod_manager = Teuchos::rcp( 
	new MSODManager<SourceType>(d_primary_set, d_global_comm, *d_plist) );

    // Build the Monte Carlo set solver.
    if ( Teuchos::is_null(d_mc_solver) )
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

    MCLS_ENSURE( !d_msod_manager.is_null() );
    MCLS_ENSURE( !d_msod_manager->localDomain().is_null() );
    MCLS_ENSURE( !d_mc_solver.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the Monte Carlo source from the provided linear problem.
 */
template<class Vector, class Matrix>
void AdjointSolverManager<Vector,Matrix>::buildMonteCarloSource()
{
    MCLS_REQUIRE( !d_global_comm.is_null() );
    MCLS_REQUIRE( !d_plist.is_null() );
    MCLS_REQUIRE( !d_msod_manager.is_null() );
    MCLS_REQUIRE( !d_msod_manager->localDomain().is_null() );
    MCLS_REQUIRE( !d_mc_solver.is_null() );

    // Set a global scope variable for the primary source.
    Teuchos::RCP<SourceType> primary_source;

    // Build the primary source in the primary set.
    if ( d_primary_set )
    {
        // Get a copy of the source vector in the operator decomposition.
        Teuchos::RCP<Vector> rhs_op_decomp = 
            MT::cloneVectorFromMatrixRows( *d_problem->getOperator() );
        VectorExport<Vector> rhs_export( 
            Teuchos::rcp_const_cast<Vector>(d_problem->getRHS()), rhs_op_decomp );
        rhs_export.doExportInsert();

        // Left precondition the source if necessary.
        if ( d_problem->isLeftPrec() )
        {
            Teuchos::RCP<Vector> temp = VT::clone(*rhs_op_decomp);
            d_problem->applyLeftPrec( *rhs_op_decomp, *temp );
            VT::update( *rhs_op_decomp,
                        Teuchos::ScalarTraits<Scalar>::zero(),
                        *temp,
                        Teuchos::ScalarTraits<Scalar>::one() );
        }

        // Scale by the Neumann relaxation parameter.
        if ( d_plist->isParameter("Neumann Relaxation") )
        {
            double omega = d_plist->get<double>("Neumann Relaxation");
            VT::scale( *rhs_op_decomp, omega );
        }

        // Build the source.
        primary_source = Teuchos::rcp( 
            new SourceType( rhs_op_decomp,
                            d_msod_manager->localDomain(),
                            d_mc_solver->rngControl(),
                            d_msod_manager->setComm(),
                            d_global_comm->getSize(),
                            d_global_comm->getRank(),
                            *d_plist ) );
    }
    d_global_comm->barrier();

    // Build the global MSOD Monte Carlo source from the primary set.
    d_msod_manager->setSource( primary_source, d_mc_solver->rngControl() );

    MCLS_ENSURE( !d_msod_manager->localSource().is_null() );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_ADJOINTSOLVERMANAGER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_AdjointSolverManager_impl.hpp
//---------------------------------------------------------------------------//

