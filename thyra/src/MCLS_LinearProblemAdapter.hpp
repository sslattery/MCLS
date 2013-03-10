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
 * \brief LinearProblem adapter for Thyra
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_LINEARPROBLEMADAPTER_HPP
#define MCLS_LINEARPROBLEMADAPTER_HPP

#include <MCLS_DBC.hpp>
#include <MCLS_LinearProblem.hpp>

#include "MCLS_MultiVectorTraits.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Describable.hpp>

#include <Thyra_MultiVectorBase.hpp>
#include <Thyra_LinearOpBase.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class LinearProblemBase
 * \brief Linear system container for A*x = b  using Thyra base classes.
 */
template<class Scalar>
class LinearProblemBase : public virtual Teuchos::Describable
{
  public:

    //@{
    //! Typedefs.
    typedef Thyra::MultiVectorBase<Scalar>             multivector_type;
    typedef Thyra::LinearOpBase<Scalar>                linear_op_type;
    //@}

    //! Constructor.
    LinearProblemBase() { /* ... */ }

    //! Destructor.
    virtual ~LinearProblemBase() { /* ... */ }

    //! Set the left-hand side.
    virtual void setLHS( const Teuchos::RCP<multivector_type>& x ) = 0;

    //! Set the right-hand side.
    virtual void setRHS( const Teuchos::RCP<const multivector_type>& b ) = 0;

    //! Set the linear operator.
    void setOperator( const Teuchos::RCP<const linear_op_type>& A ) 
    { b_A = A; }

    //! Get the linear operator.
    Teuchos::RCP<const linear_op_type> getOperator() const 
    { return b_A; }

  private:

    // Linear operator base class.
    Teuchos::RCP<const linear_op_type> b_A;
};

//---------------------------------------------------------------------------//
/*!
 * \class LinearProblemAdapter
 * \brief Linear system container for A*x = b.
 *
 * This container holds a blocked linear system and provides access to the
 * individual linear systems contained within for MCLS solves.
 */
template<class MultiVector, class Matrix>
class LinearProblemAdapter : public LinearProblemBase<
    typename MultiVectorTraits<MultiVector>::scalar_type>
{
  public:

    //@{
    //! Typedefs.
    typedef MultiVector                                  multivector_type;
    typedef MultiVectorTraits<MultiVector>               MVT;
    typedef typename MVT::vector_type                    Vector;
    typedef Matrix                                       matrix_type;
    typedef LinearProblemBase<typename MVT::scalar_type> Base;
    //@}

    //! Default constructor.
    LinearProblemAdapter()
	: d_num_problems(0)
    { /* ... */ }

    //! Constructor.
    LinearProblemAdapter( const Teuchos::RCP<const Matrix>& A,
			  const Teuchos::RCP<MultiVector>& x,
			  const Teuchos::RCP<const MultiVector>& b,
			  const int num_problems )
	: d_A( A )
	, d_x( x )
	, d_b( b )
	, d_num_problems( num_problems )
    { /* ... */ }

    //! Destructor.
    ~LinearProblemAdapter() { /* ... */ }

    //! Get a subproblem given a LHS/RHS id.
    Teuchos::RCP<LinearProblem<Vector,Matrix> > getSubProblem( const int id )
    {
	MCLS_REQUIRE( id < MVT::getNumVectors(*d_x) );
	MCLS_REQUIRE( id < MVT::getNumVectors(*d_b) );

	Teuchos::RCP<Vector> vector_x = MVT::getVectorNonConst( *d_x, id );
	Teuchos::RCP<const Vector> vector_b = MVT::getVector( *d_b, id );

	Teuchos::RCP<LinearProblem<Vector,Matrix> > lp = Teuchos::rcp(
	    new LinearProblem<Vector,Matrix>(d_A, vector_x, vector_b) );

	lp->setLeftPrec( d_PL );
	lp->setRightPrec( d_PR );

	return lp;
    }

    //! Get the number of LHS/RHS in the problem.
    int getNumSubProblems()
    { 
	if ( d_num_problems == 0 );
	{
	    MCLS_CHECK( Teuchos::nonnull(d_x) );
	    d_num_problems = MVT::getNumVectors(*d_x);
	}

	return d_num_problems;
    }  

    //! Set the number of LHS/RHS in the problem.
    void setNumSubProblems( const int num_problems ) const 
    { d_num_problems = num_problems; }  

    //! Set the linear operator.
    void setOperator( const Teuchos::RCP<const Matrix>& A ) { d_A = A; }

    //! Set the left-hand side.
    void setLHS( const Teuchos::RCP<MultiVector>& x ) { d_x = x; }

    //! Overload for Thyra operator.
    void setLHS( const Teuchos::RCP<typename Base::multivector_type>& x )
    { 
	d_x = MVT::getMultiVectorFromThyra( x );
	MCLS_ENSURE( Teuchos::nonnull(d_x) );
    }

    //! Set the right-hand side.
    void setRHS( const Teuchos::RCP<const MultiVector>& b ) { d_b = b; }

    //! Overload for Thyra operator.
    void setRHS( const Teuchos::RCP<const typename Base::multivector_type>& b )
    { 
	d_b = MVT::getConstMultiVectorFromThyra( b );
	MCLS_ENSURE( Teuchos::nonnull(d_b) );
    }

    //! Set the left preconditioner.
    void setLeftPrec( const Teuchos::RCP<const Matrix>& PL ) { d_PL = PL; }

    //! Set the right preconditioner.
    void setRightPrec( const Teuchos::RCP<const Matrix>& PR )  { d_PR = PR; }

    //! Get the linear operator.
    Teuchos::RCP<const Matrix> getOperator() const { return d_A; }

    //! Get the left-hand side.
    Teuchos::RCP<MultiVector> getLHS() const { return d_x; }

    //! Get the right-hand side.
    Teuchos::RCP<const MultiVector> getRHS() const { return d_b; }

    //! Get the left preconditioner.
    Teuchos::RCP<const Matrix> getLeftPrec() const { return d_PL; }

    //! Get the right preconditioner.
    Teuchos::RCP<const Matrix> getRightPrec() const { return d_PR; }

  private:

    // Linear operator.
    Teuchos::RCP<const Matrix> d_A;

    // Left-hand side.
    Teuchos::RCP<MultiVector> d_x;

    // Right-hand side.
    Teuchos::RCP<const MultiVector> d_b;

    // Left preconditioner.
    Teuchos::RCP<const Matrix> d_PL;

    // Right preconditioner.
    Teuchos::RCP<const Matrix> d_PR;

    // Number of LHS/RHS in the problem.
    int d_num_problems;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_LINEARPROBLEMADAPTER_HPP

//---------------------------------------------------------------------------//
// end MCLS_LinearProblemAdapter.hpp
// ---------------------------------------------------------------------------//

