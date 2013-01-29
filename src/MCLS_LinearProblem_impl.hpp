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
 * \file MCLS_LinearProblem_impl.hpp
 * \author Stuart R. Slattery
 * \brief Linear problem implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_LINEARPROBLEM_IMPL_HPP
#define MCLS_LINEARPROBLEM_IMPL_HPP

#include "MCLS_DBC.hpp"

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector, class Matrix>
LinearProblem<Vector,Matrix>::LinearProblem( 
    const Teuchos::RCP<const Matrix>& A,
    const Teuchos::RCP<Vector>& x,
    const Teuchos::RCP<const Vector>& b )
    : d_A( A )
    , d_x( x )
    , d_b( b )
    , d_r( VT::clone( *d_x ) )
{
    d_status = true;

    Ensure( !d_A.is_null() );
    Ensure( !d_x.is_null() );
    Ensure( !d_b.is_null() );
    Ensure( !d_r.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Destructor.
 */
template<class Vector, class Matrix>
LinearProblem<Vector,Matrix>::~LinearProblem()
{ /* ... */ }

//---------------------------------------------------------------------------//
/*!
 * \brief Set the linear operator
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::setOperator( 
    const Teuchos::RCP<const Matrix>& A )
{
    Require( !A.is_null() );

    d_A = A;
    d_status = false;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the left-hand side
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::setLHS( const Teuchos::RCP<Vector>& x )
{
    Require( !x.is_null() );

    d_x = x;
    d_status = false;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the right-hand side
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::setRHS( const Teuchos::RCP<const Vector>& b )
{
    Require( !b.is_null() );

    d_b = b;
    d_status = false;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Apply the linear operator to a vector.
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::applyOperator( const Vector& x, Vector& y )
{
    MT::apply( *d_A, x, y );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Update the residual.
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::updateResidual()
{
    applyOperator( *d_x, *d_r );
    VT::update( *d_r, -1.0, *d_b, 1.0 );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_LINEARPROBLEM_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_LinearProblem_impl.hpp
// ---------------------------------------------------------------------------//

