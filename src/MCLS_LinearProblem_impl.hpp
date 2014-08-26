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

#include <Teuchos_ScalarTraits.hpp>

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
    , d_r( MT::cloneVectorFromMatrixRows(*d_A) )
    , d_rp( MT::cloneVectorFromMatrixRows(*d_A) )
    , d_mm_timer( Teuchos::TimeMonitor::getNewCounter("MM Multiply") )
{
    MCLS_ENSURE( Teuchos::nonnull(d_A) );
    MCLS_ENSURE( Teuchos::nonnull(d_x) );
    MCLS_ENSURE( Teuchos::nonnull(d_b) );
    MCLS_ENSURE( Teuchos::nonnull(d_r) );
    MCLS_ENSURE( Teuchos::nonnull(d_rp) );
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
 * \brief Set the linear operator.
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::setOperator( 
    const Teuchos::RCP<const Matrix>& A )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );
    d_A = A;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the left-hand side.
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::setLHS( const Teuchos::RCP<Vector>& x )
{
    MCLS_REQUIRE( Teuchos::nonnull(x) );
    d_x = x;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the right-hand side.
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::setRHS( const Teuchos::RCP<const Vector>& b )
{
    MCLS_REQUIRE( Teuchos::nonnull(b) );
    d_b = b;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the left preconditioner.
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::setLeftPrec( 
    const Teuchos::RCP<const Matrix>& PL )
{
    MCLS_REQUIRE( Teuchos::nonnull(PL) );
    d_PL = PL;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the right preconditioner.
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::setRightPrec( 
    const Teuchos::RCP<const Matrix>& PR )
{
    MCLS_REQUIRE( Teuchos::nonnull(PR) );
    d_PR = PR;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the composite linear operator.
 */
template<class Vector, class Matrix>
Teuchos::RCP<const Matrix> 
LinearProblem<Vector,Matrix>::getCompositeOperator() const
{
    Teuchos::TimeMonitor mm_monitor( *d_mm_timer );

    const bool left_prec = Teuchos::nonnull( d_PL );
    const bool right_prec = Teuchos::nonnull( d_PR );

    Teuchos::RCP<Matrix> composite;

    if ( left_prec && right_prec )
    {
        Teuchos::RCP<Matrix> temp = MT::clone( *d_A );
	MT::multiply( d_A, false, d_PR, false, temp );
        composite = MT::clone( *d_PL );
	MT::multiply( d_PL, false, temp, false, composite );
    }
    else if ( left_prec )
    {
        composite = MT::clone( *d_PL );
	MT::multiply( d_PL, false, d_A, false, composite );
    }
    else if ( right_prec )
    {
        composite = MT::clone( *d_A );
	MT::multiply( d_A, false, d_PR, false, composite );
    }
    else
    {
	composite = Teuchos::rcp_const_cast<Matrix>( d_A );
    }

    return composite;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the transposed composite linear operator.
 */
template<class Vector, class Matrix>
Teuchos::RCP<const Matrix> 
LinearProblem<Vector,Matrix>::getTransposeCompositeOperator() const
{
    Teuchos::TimeMonitor mm_monitor( *d_mm_timer );

    const bool left_prec = Teuchos::nonnull( d_PL );
    const bool right_prec = Teuchos::nonnull( d_PR );

    Teuchos::RCP<Matrix> composite;

    if ( left_prec && right_prec )
    {
	Teuchos::RCP<Matrix> temp;
	{
	    Teuchos::RCP<Matrix> PL_T = MT::copyTranspose( *d_PL );
	    Teuchos::RCP<Matrix> A_T = MT::copyTranspose( *d_A );
	    temp = MT::clone( *A_T );
	    MT::multiply( A_T, false, PL_T, false, temp );
	}
	Teuchos::RCP<Matrix> PR_T = MT::copyTranspose( *d_PR );
	composite = MT::clone( *PR_T );        
	MT::multiply( PR_T, false, temp, false, composite );
    }
    else if ( right_prec )
    {
	Teuchos::RCP<Matrix> A_T = MT::copyTranspose( *d_A );
	Teuchos::RCP<Matrix> PR_T = MT::copyTranspose( *d_PR );
	composite = MT::clone( *PR_T );
	MT::multiply( PR_T, false, A_T, false, composite );
    }
    else if ( left_prec )
    {
	Teuchos::RCP<Matrix> A_T = MT::copyTranspose( *d_A );
	composite = MT::clone( *A_T );
	Teuchos::RCP<Matrix> PL_T = MT::copyTranspose( *d_PL );
	MT::multiply( A_T, false, PL_T, false, composite );
    }
    else
    {
	composite = MT::copyTranspose( *d_A );
    }

    return composite;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Update the solution vector with the provided update vector.
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::updateSolution( 
    const Teuchos::RCP<Vector>& update )
{
    const bool right_prec = Teuchos::nonnull( d_PR );

    if ( right_prec )
    {
	Teuchos::RCP<Vector> prec_update = VT::clone(*update);
	MT::apply( *d_PR, *update, *prec_update );
	VT::update( *d_x, Teuchos::ScalarTraits<Scalar>::one(),
		    *prec_update, Teuchos::ScalarTraits<Scalar>::one() );
    }
    else
    {
	VT::update( *d_x, Teuchos::ScalarTraits<Scalar>::one(),
		    *update, Teuchos::ScalarTraits<Scalar>::one() );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Apply the composite linear operator to a vector.
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::apply( const Vector& x, Vector& y )
{
    const bool left_prec = Teuchos::nonnull( d_PL );
    const bool right_prec = Teuchos::nonnull( d_PR );

    Teuchos::RCP<Vector> temp = 
	( left_prec || right_prec ) ? VT::clone(y) : Teuchos::null;

    if ( !left_prec && !right_prec )
    {
	MT::apply( *d_A, x, y );
    }

    else if ( left_prec && right_prec )
    {
	MT::apply( *d_PR, x, y );
	MT::apply( *d_A, y, *temp );
	MT::apply( *d_PL, *temp, y );
    }
    else if ( left_prec )
    {
	MT::apply( *d_A, x, *temp );
	MT::apply( *d_PL, *temp, y );
    }
    else
    {
	MT::apply( *d_PR, x, *temp );
	MT::apply( *d_A, *temp, y );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Apply the transpose composite linear operator to a vector.
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::applyTranspose( const Vector& x, Vector& y )
{
    const bool left_prec = Teuchos::nonnull( d_PL );
    const bool right_prec = Teuchos::nonnull( d_PR );

    Teuchos::RCP<Vector> temp = 
	( left_prec || right_prec ) ? VT::clone(y) : Teuchos::null;

    if ( !left_prec && !right_prec )
    {
	MT::applyTranspose( *d_A, x, y );
    }

    else if ( left_prec && right_prec )
    {
	MT::applyTranspose( *d_PL, x, y );
	MT::applyTranspose( *d_A, y, *temp );
	MT::applyTranspose( *d_PR, *temp, y );
    }
    else if ( left_prec )
    {
	MT::applyTranspose( *d_PL, x, *temp );
	MT::applyTranspose( *d_A, *temp, y );
    }
    else
    {
	MT::applyTranspose( *d_A, x, *temp );
	MT::applyTranspose( *d_PR, *temp, y );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Apply the base linear operator to a vector.
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::applyOp( const Vector& x, Vector& y )
{
    if ( Teuchos::nonnull(d_A) )
    {
	MT::apply( *d_A, x, y );
    }
    else
    {
	VT::update( y, Teuchos::ScalarTraits<Scalar>::zero(), 
		    x, Teuchos::ScalarTraits<Scalar>::one() );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Apply the left preconditioner to a vector.
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::applyLeftPrec( const Vector& x, Vector& y )
{
    if ( Teuchos::nonnull(d_PL) )
    {
	MT::apply( *d_PL, x, y );
    }
    else
    {
	VT::update( y, Teuchos::ScalarTraits<Scalar>::zero(), 
		    x, Teuchos::ScalarTraits<Scalar>::one() );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Apply the right preconditioner to a vector.
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::applyRightPrec( const Vector& x, Vector& y )
{
    if ( Teuchos::nonnull(d_PR) )
    {
	MT::apply( *d_PR, x, y );
    }
    else
    {
	VT::update( y, Teuchos::ScalarTraits<Scalar>::zero(), 
		    x, Teuchos::ScalarTraits<Scalar>::one() );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Update the residual.
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::updateResidual()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_A) );
    MCLS_REQUIRE( Teuchos::nonnull(d_x) );
    MCLS_REQUIRE( Teuchos::nonnull(d_b) );

    MT::apply( *d_A, *d_x, *d_r );
    VT::update( *d_r, -Teuchos::ScalarTraits<Scalar>::one(), 
		*d_b, Teuchos::ScalarTraits<Scalar>::one() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Update the preconditioned residual. Preconditioning will be applied
 * if preconditioners are present. 
 */
template<class Vector, class Matrix>
void LinearProblem<Vector,Matrix>::updatePrecResidual()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_A) );
    MCLS_REQUIRE( Teuchos::nonnull(d_x) );
    MCLS_REQUIRE( Teuchos::nonnull(d_b) );

    // Apply right preconditioning if necessary.
    if ( Teuchos::nonnull(d_PR) )
    {
        Teuchos::RCP<Vector> temp = VT::clone(*d_rp);
        MT::apply( *d_PR, *d_x, *temp);
        MT::apply( *d_A, *temp, *d_rp );
    }
    else
    {
        MT::apply( *d_A, *d_x, *d_rp );
    }

    VT::update( *d_rp, -Teuchos::ScalarTraits<Scalar>::one(), 
		*d_b, Teuchos::ScalarTraits<Scalar>::one() );

    // Apply left preconditioning if necessary.
    if ( Teuchos::nonnull(d_PL) )
    {
        Teuchos::RCP<Vector> temp = VT::deepCopy(*d_rp);
	MT::apply( *d_PL, *temp, *d_rp );
    }
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_LINEARPROBLEM_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_LinearProblem_impl.hpp
// ---------------------------------------------------------------------------//

