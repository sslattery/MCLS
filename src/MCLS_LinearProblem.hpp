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
 * \file MCLS_LinearProblem.hpp
 * \author Stuart R. Slattery
 * \brief Linear Problem declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_LINEARPROBLEM_HPP
#define MCLS_LINEARPROBLEM_HPP

#include "MCLS_config.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_MatrixTraits.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Time.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class LinearProblem
 * \brief Linear system container for A*x = b.
 */
template<class Vector, class Matrix>
class LinearProblem
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                      vector_type;
    typedef VectorTraits<Vector>                        VT;
    typedef typename VT::scalar_type                    Scalar;
    typedef Matrix                                      matrix_type;
    typedef MatrixTraits<Vector,Matrix>                 MT;
    //@}

    // Constructor.
    LinearProblem( const Teuchos::RCP<const Matrix>& A,
		   const Teuchos::RCP<Vector>& x,
		   const Teuchos::RCP<const Vector>& b );

    // Destructor.
    ~LinearProblem();

    // Set the linear operator.
    void setOperator( const Teuchos::RCP<const Matrix>& A );

    // Set the left-hand side.
    void setLHS( const Teuchos::RCP<Vector>& x );

    // Set the right-hand side.
    void setRHS( const Teuchos::RCP<const Vector>& b );

    // Set the left preconditioner.
    void setLeftPrec( const Teuchos::RCP<const Matrix>& PL );

    // Set the right preconditioner.
    void setRightPrec( const Teuchos::RCP<const Matrix>& PR );

    //! Get the linear operator.
    Teuchos::RCP<const Matrix> getOperator() const { return d_A; }

    //! Get the left-hand side.
    Teuchos::RCP<Vector> getLHS() const { return d_x; }

    //! Get the right-hand side.
    Teuchos::RCP<const Vector> getRHS() const { return d_b; }

    //! Get the left preconditioner.
    Teuchos::RCP<const Matrix> getLeftPrec() const { return d_PL; }

    //! Get the right preconditioner.
    Teuchos::RCP<const Matrix> getRightPrec() const { return d_PR; }

    // Get the composite linear operator.
    Teuchos::RCP<const Matrix> getCompositeOperator( 
	const double threshold = 0.0 ) const;

    // Get the transposed composite linear operator.
    Teuchos::RCP<const Matrix> getTransposeCompositeOperator(
	const double threshold = 0.0 ) const;

    //! Get the residual vector. 
    Teuchos::RCP<const Vector> getResidual() const { return d_r; }

    //! Get the residual vector. This will be preconditioned if
    //! preconditioners are present.
    Teuchos::RCP<const Vector> getPrecResidual() const { return d_rp; }
    Teuchos::RCP<Vector> getPrecResidual() { return d_rp; }
    
    //! Determine if the linear system is left preconditioned.
    bool isLeftPrec() const { return Teuchos::nonnull(d_PL); }

    //! Determine if the linear system is right preconditioned.
    bool isRightPrec() const { return Teuchos::nonnull(d_PR); }

    // Update the solution vector with a provided update vector.
    void updateSolution( const Teuchos::RCP<Vector>& update );

    // Apply the composite linear operator to a vector.
    void apply( const Vector& x, Vector& y );

    // Apply the transpose composite linear operator to a vector.
    void applyTranspose( const Vector& x, Vector& y );

    // Apply the base linear operator to a vector.
    void applyOp( const Vector& x, Vector& y );

    // Apply the left preconditioner to a vector.
    void applyLeftPrec( const Vector& x, Vector& y );

    // Apply the right preconditioner to a vector.
    void applyRightPrec( const Vector& x, Vector& y );

    // Update the residual. 
    void updateResidual();

    // Update the preconditioned residual. Preconditioning will be applied if
    // preconditioners are present. The unpreconditioned residual will be
    // updated as well.
    void updatePrecResidual();

  private:

    // Linear operator.
    Teuchos::RCP<const Matrix> d_A;

    // Left-hand side.
    Teuchos::RCP<Vector> d_x;

    // Right-hand side.
    Teuchos::RCP<const Vector> d_b;

    // Left preconditioner.
    Teuchos::RCP<const Matrix> d_PL;

    // Right preconditioner.
    Teuchos::RCP<const Matrix> d_PR;

    // Residual r = b - A*x.
    Teuchos::RCP<Vector> d_r;

    // Preconditioned residual rp = PL*(b - A*PR*x).
    Teuchos::RCP<Vector> d_rp;

#if HAVE_MCLS_TIMERS
    // Matrix-matrix multiply timer.
    Teuchos::RCP<Teuchos::Time> d_mm_timer;

    // Matrix-vector multiply timer.
    Teuchos::RCP<Teuchos::Time> d_mv_timer;
#endif
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_LinearProblem_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_LINEARPROBLEM_HPP

//---------------------------------------------------------------------------//
// end MCLS_LinearProblem.hpp
// ---------------------------------------------------------------------------//

