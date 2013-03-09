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
 * \file MCLS_TpetraPointJacobiPreconditioner.hpp
 * \author Stuart R. Slattery
 * \brief Tpetra::Vector Export.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_TPETRAPOINTJACOBI_HPP
#define MCLS_TPETRAPOINTJACOBI_HPP

#include <MCLS_DBC.hpp>

#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ArrayRCP.hpp>

#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class TpetraPointJacobiPreconditioner
 * \brief Point-Jacobi preconditioner for Tpetra::CrsMatrix
 */
template<class Scalar, class LO, class GO>
class TpetraPointJacobiPreconditioner
{
  public:

    //@{
    //! Typedefs.
    typedef Tpetra::Vector<Scalar,LO,GO>            vector_type;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO>         matrix_type;
    //@}

    /*!
     * \brief Constructor.
     */
    TpetraPointJacobiPreconditioner() { /* ... */ }

    /*!
     * \brief Destructor.
     */
    ~TpetraPointJacobiPreconditioner() { /* ... */ }

    /*!
     * \brief Set the operator with the preconditioner.
     */
    void setOperator( const Teuchos::RCP<const matrix_type>& A )
    {
	MCLS_REQUIRE( Teuchos::nonnull(A) );
	d_A = A;
    }

    /*!
     * \brief Build the preconditioner.
     */
    void buildPreconditioner()
    {
	MCLS_REQUIRE( d_A->isFillComplete() );

	// Create the preconditioner.
	d_preconditioner = 
	    Tpetra::createCrsMatrix<Scalar,LO,GO>( d_A->getRowMap() );

	// Compute the inverse of the diagonal.
	Teuchos::RCP<vector_type> diagonal = 
	    Tpetra::createVector<Scalar,LO,GO>( d_A->getRowMap() );
	d_A->getLocalDiagCopy( *diagonal );
	diagonal->reciprocal( *diagonal );
	Teuchos::ArrayRCP<const Scalar> diagonal_data = diagonal->getData();

	// Build a matrix from the diagonal vector.
	Teuchos::ArrayView<const GO> rows = 
	    d_preconditioner->getRowMap()->getNodeElementsList();
	Teuchos::ArrayView<const GO>::const_iterator row_it;
	Teuchos::Array<GO> col(1);
	LO local_row = 0;
	for ( row_it = rows.begin(); row_it != rows.end(); ++row_it )
	{
	    col[0] = *row_it;
	    d_preconditioner->insertGlobalValues( 
		*row_it, col(), diagonal_data( local_row, 1 ) );
	    ++local_row;
	}

	d_preconditioner->fillComplete();
	
	MCLS_ENSURE( Teuchos::nonull(d_preconditioner) );
	MCLS_ENSURE( d_preconditioner->isFillComplete() );
    }

    /*!
     * \brief Get the preconditioner.
     */
    Teuchos::RCP<const matrix_type> getPreconditioner() const
    { return d_preconditioner; }

  private:

    // Original operator.
    Teuchos::RCP<const matrix_type> d_A;

    // Preconditioner (M^-1)
    Teuchos::RCP<matrix_type> d_preconditioner;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_TPETRAPOINTJACOBI_HPP

//---------------------------------------------------------------------------//
// end MCLS_TpetraPointJacobiPreconditioner.hpp
//---------------------------------------------------------------------------//
