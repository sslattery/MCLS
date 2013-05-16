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
 * \file MCLS_MinimalResidualIteration_impl.hpp
 * \author Stuart R. Slattery
 * \brief Minimal Residual iteration implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MINIMALRESIDUALITERATION_IMPL_HPP
#define MCLS_MINIMALRESIDUALITERATION_IMPL_HPP

#include "MCLS_DBC.hpp"

#include <Teuchos_ScalarTraits.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Default constructor. setProblem() must be called before solve().
 */
template<class Vector, class Matrix>
MinimalResidualIteration<Vector,Matrix>::MinimalResidualIteration()
{ /* ... */ }

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector, class Matrix>
MinimalResidualIteration<Vector,Matrix>::MinimalResidualIteration( 
    const Teuchos::RCP<LinearProblemType>& problem )
    : d_problem( problem )
    , d_p( VT::clone(*problem->getPrecResidual()) )
{ 
    MCLS_ENSURE( Teuchos::nonnull(d_problem) );
    MCLS_ENSURE( Teuchos::nonnull(d_p) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the valid parameters for this manager.
 */
template<class Vector, class Matrix>
Teuchos::RCP<const Teuchos::ParameterList> 
MinimalResidualIteration<Vector,Matrix>::getValidParameters() const
{
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    return plist;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the linear problem with the manager.
 */
template<class Vector, class Matrix>
void MinimalResidualIteration<Vector,Matrix>::setProblem( 
    const Teuchos::RCP<LinearProblem<Vector,Matrix> >& problem )
{
    MCLS_REQUIRE( Teuchos::nonnull(problem) );
    d_problem = problem;
    d_p = VT::clone( *d_problem->getPrecResidual() );
    MCLS_ENSURE( Teuchos::nonnull(d_p) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the parameters for the manager. The manager will modify this
 * list with default parameters that are not defined.
 */
template<class Vector, class Matrix>
void MinimalResidualIteration<Vector,Matrix>::setParameters( 
    const Teuchos::RCP<Teuchos::ParameterList>& params )
{ /* ... */ }

//---------------------------------------------------------------------------//
/*!
 * \brief Do a single fixed point iteration. Must update the residual.
 */
template<class Vector, class Matrix>
void MinimalResidualIteration<Vector,Matrix>::doOneIteration()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_problem) );
    MCLS_REQUIRE( Teuchos::nonnull(d_p) );

    // Build p = A*r.
    d_problem->apply( *d_problem->getPrecResidual(), *d_p );

    // Petrov-Galerkin condition.
    Scalar alpha = VT::dot( *d_problem->getPrecResidual(), 
                            *d_problem->getPrecResidual() ) /
                   VT::dot( *d_p, *d_p );

    // Fixed point update.
    VT::update( *d_problem->getLHS(), 
                Teuchos::ScalarTraits<Scalar>::one(),
                *d_problem->getPrecResidual(),
                alpha );

    // Residual update.
    VT::update( *d_problem->getPrecResidual(),
                Teuchos::ScalarTraits<Scalar>::one(),
                *d_p,
                -alpha );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_MINIMALRESIDUALITERATION_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_MinimalResidualIteration_impl.hpp
//---------------------------------------------------------------------------//

