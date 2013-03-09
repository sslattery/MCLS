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
 * \file MCLS_EpetraPointJacobiPreconditioner.hpp
 * \author Stuart R. Slattery
 * \brief Point Jacobi preconditioning for Epetra.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_EPETRAPOINTJACOBI_HPP
#define MCLS_EPETRAPOINTJACOBI_HPP

#include <MCLS_DBC.hpp>

#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ArrayRCP.hpp>

#include <Epetra_Vector.h>
#include <Epetra_CrsMatrix.h>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class EpetraPointJacobiPreconditioner
 * \brief Point-Jacobi preconditioner for Epetra_CrsMatrix
 */
class EpetraPointJacobiPreconditioner
{
  public:

    //@{
    //! Typedefs.
    typedef Epetra_Vector                           vector_type;
    typedef Epetra_CrsMatrix                        matrix_type;
    //@}

    /*!
     * \brief Constructor.
     */
    EpetraPointJacobiPreconditioner() { /* ... */ }

    /*!
     * \brief Destructor.
     */
    ~EpetraPointJacobiPreconditioner() { /* ... */ }

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
	MCLS_REQUIRE( Teuchos::nonnull(d_A) );
	MCLS_REQUIRE( d_A->Filled() );

	// Create the preconditioner.
	d_preconditioner = Teuchos::rcp( 
	    new Epetra_CrsMatrix(Copy,d_A->RowMatrixRowMap(),0) );
	    
	// Compute the inverse of the diagonal.
	Teuchos::RCP<vector_type> diagonal = 
	    Teuchos::rcp( new vector_type( d_A->RowMatrixRowMap() ) );

	d_A->ExtractDiagonalCopy( *diagonal );
	diagonal->Reciprocal( *diagonal );

	// Build a matrix from the diagonal vector.
	Teuchos::ArrayView<const GO> rows = 
	    d_preconditioner->getRowMap()->getNodeElementsList();
	Teuchos::ArrayView<const GO>::const_iterator row_it;
	Teuchos::Array<GO> col(1);
	LO local_row = 0;
	for ( row_it = rows.begin(); row_it != rows.end(); ++row_it )
	{
	    col[0] = *row_it;
	    d_preconditioner->InsertGlobalValues( 
		*row_it, 1, &diagonal_data[local_row], col->getRawPtr() );
	    ++local_row;
	}

	d_preconditioner->FillComplete();
	
	MCLS_ENSURE( Teuchos::nonull(d_preconditioner) );
	MCLS_ENSURE( d_preconditioner->Filled() );
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

#endif // end MCLS_EPETRAPOINTJACOBI_HPP

//---------------------------------------------------------------------------//
// end MCLS_EpetraPointJacobiPreconditioner.hpp
//---------------------------------------------------------------------------//
