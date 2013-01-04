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
 * \file MCLS_TpetraCrsMatrixAdapter.hpp
 * \author Stuart R. Slattery
 * \brief Tpetra::CrsMatrix Adapter.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_TPETRACRSMATRIXADAPTER_HPP
#define MCLS_TPETRACRSMATRIXADAPTER_HPP

#include <MCLS_Assertion.hpp>
#include <MCLS_MatrixTraits.hpp>

#include <Teuchos_as.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Array.hpp>

#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class VectorTraits
 * \brief Traits specialization for Tpetra::Vector.
 */
template<class Scalar, class LO, class GO>
class MatrixTraits<Scalar,LO,GO,Tpetra::Vector<Scalar,LO,GO>,
		   Tpetra::CrsMatrix<Scalar,LO,GO> >
{
  public:

    //@{
    //! Typedefs.
    typedef typename Tpetra::CrsMatrix<Scalar,LO,GO>     matrix_type;
    typedef typename Tpetra::Vector<Scalar,LO,GO>        vector_type;
    typedef typename vector_type::scalar_type            scalar_type;
    typedef typename vector_type::local_ordinal_type     local_ordinal_type;
    typedef typename vector_type::global_ordinal_type    global_ordinal_type;
    //@}

    /*!
     * \brief Create a reference-counted pointer to a new empty matrix with
     * the same parallel distribution as the given matrix.
     */
    static Teuchos::RCP<matrix_type> clone( const matrix_type& matrix )
    { 
	return Tpetra::createCrsMatrix<Scalar,LO,GO>( matrix.getMap() );
    }

    /*!
     * \brief Create a reference-counted pointer to a new empty vector from a
     * matrix to give the vector the same parallel distribution as the
     * matrix parallel row distribution.
     */
    static Teuchos::RCP<vector_type> 
    cloneVectorFromMatrixRows( const matrix_type& matrix )
    { 
	return Tpetra::createVector<Scalar,LO,GO>( matrix.getRowMap() );
    }

    /*!
     * \brief Create a reference-counted pointer to a new empty vector from a
     * matrix to give the vector the same parallel distribution as the
     * matrix parallel column distribution.
     */
    static Teuchos::RCP<vector_type> 
    cloneVectorFromMatrixCols( const matrix_type& matrix )
    { 
	testPrecondition( matrix.isFillComplete() );
	return Tpetra::createVector<Scalar,LO,GO>( matrix.getColMap() );
    }

    /*!
     * \brief Get the communicator.
     */
    static const Teuchos::RCP<const Teuchos::Comm<int> >&
    getComm( const matrix_type& matrix )
    {
	return matrix.getComm();
    }

    /*!
     * \brief Get the global number of rows.
     */
    static GO getGlobalNumRows( const matrix_type& matrix )
    { 
	return Teuchos::as<GO>( matrix.getGlobalNumRows() );
    }

    /*!
     * \brief Get the local number of rows.
     */
    static LO getLocalNumRows( const matrix_type& matrix )
    {
	return Teuchos::as<LO>( matrix.getRowMap()->getNodeNumElements() );
    }

    /*!
     * \brief Get the maximum number of entries in a row globally.
     */
    static GO getGlobalMaxNumRowEntries( const matrix_type& matrix )
    {
	return Teuchos::as<GO>( matrix.getGlobalMaxNumRowEntries() );
    }

    /*!
     * \brief Given a local row on-process, provide the global ordinal.
     */
    static GO getGlobalRow( const matrix_type& matrix, const LO& local_row )
    { 
	return matrix.getRowMap()->getGlobalElement( local_row );
    }

    /*!
     * \brief Given a global row on-process, provide the local ordinal.
     */
    static LO getLocalRow( const matrix_type& matrix, const GO& global_row )
    { 
	return matrix.getRowMap()->getLocalElement( global_row );
    }

    /*!
     * \brief Given a local col on-process, provide the global ordinal.
     */
    static GO getGlobalCol( const matrix_type& matrix, const LO& local_col )
    {
	testPrecondition( matrix.isFillComplete() );
	return matrix.getColMap()->getGlobalElement( local_col );
    }

    /*!
     * \brief Given a global col on-process, provide the local ordinal.
     */
    static LO getLocalCol( const matrix_type& matrix, const GO& global_col )
    {
	testPrecondition( matrix.isFillComplete() );
	return matrix.getColMap()->getLocalElement( global_col );
    }

    /*!
     * \brief Determine whether or not a given global row is on-process.
     */
    static bool isGlobalRow( const matrix_type& matrix, const GO& global_row )
    {
	return matrix.getRowMap()->isNodeGlobalElement( global_row );
    }

    /*!
     * \brief Determine whether or not a given local row is on-process.
     */
    static bool isLocalRow( const matrix_type& matrix, const LO& local_row )
    { 
	return matrix.getRowMap()->isNodeLocalElement( local_row );
    }

    /*!
     * \brief Determine whether or not a given global col is on-process.
     */
    static bool isGlobalCol( const matrix_type& matrix, const GO& global_col )
    { 
	testPrecondition( matrix.isFillComplete() );
	return matrix.getColMap()->isNodeGlobalElement( global_col );
    }

    /*!
     * \brief Determine whether or not a given local col is on-process.
     */
    static bool isLocalCol( const matrix_type& matrix, const LO& local_col )
    { 
	testPrecondition( matrix.isFillComplete() );
	return matrix.getColMap()->isNodeLocalElement( local_col );
    }

    /*!
     * \brief Get a view of a global row.
     */
    static void getGlobalRowView( const matrix_type& matrix,
				  const GO& global_row, 
				  Teuchos::ArrayView<const GO> &indices,
				  Teuchos::ArrayView<const Scalar> &values )
    {
	testPrecondition( !matrix.isFillComplete() );
	matrix.getGlobalRowView( global_row, indices, values );
    }

    /*!
     * \brief Get a view of a local row.
     */
    static void getLocalRowView( const matrix_type& matrix,
				 const LO& local_row, 
				 Teuchos::ArrayView<const LO> &indices,
				 Teuchos::ArrayView<const Scalar> &values )
    {
	testPrecondition( matrix.isFillComplete() );
	matrix.getLocalRowView( local_row, indices, values );
    }

    /*!
     * \brief Get a copy of the local diagonal of the matrix.
     */
    static void getLocalDiagCopy( const matrix_type& matrix, 
				  vector_type& vector )
    { 
	matrix.getLocalDiagCopy( vector );
    }

    /*!
     * \brief Apply the row matrix to a vector. A*x = y.
     */
    static void apply( const matrix_type& A, 
		       const vector_type& x, 
		       vector_type& y )
    {
	A.apply( x, y );
    }

    /*!
     * \brief Create a reference-counted pointer to a new matrix by
     * subtracting a matrix from the identity matrix. H = I - A.
     */
    static Teuchos::RCP<matrix_type>
    subtractMatrixFromIdentity( const matrix_type& matrix )
    { 
	testPrecondition( matrix.isFillComplete() );

	Teuchos::RCP<const Tpetra::Map<LO,GO> > row_map =
	    matrix.getRowMap();
	Teuchos::RCP<const Tpetra::Map<LO,GO> > col_map =
	    matrix.getColMap();

	Teuchos::RCP<matrix_type> i_minus_a = Teuchos::rcp(
	    new matrix_type( 
		row_map, col_map, matrix.getGlobalMaxNumRowEntries() ) );

	Teuchos::ArrayView<const LO> local_cols;
	typename Teuchos::ArrayView<const LO>::const_iterator 
	    local_cols_it;
	Teuchos::ArrayView<const Scalar> local_values;
	typename Teuchos::ArrayView<const Scalar>::const_iterator 
	    local_values_it;
	Teuchos::Array<Scalar> i_minus_a_vals;
	typename Teuchos::Array<Scalar>::iterator i_minus_a_vals_it;

	for ( LO local_row = row_map->getMinLocalIndex();
	      local_row <= row_map->getMaxLocalIndex();
	      ++local_row )
	{
	    matrix.getLocalRowView(
		local_row, local_cols, local_values );

	    i_minus_a_vals.resize( local_values.size() );

	    for ( local_cols_it = local_cols.begin(),
		  local_values_it = local_values.begin(),
		i_minus_a_vals_it = i_minus_a_vals.begin();
		  local_cols_it != local_cols.end();
		  ++local_cols_it, ++local_values_it, ++i_minus_a_vals_it )
	    {
		if ( row_map->getGlobalElement( local_row ) == 
		     col_map->getGlobalElement( *local_cols_it ) )
		{
		    *i_minus_a_vals_it = 1.0 - (*local_values_it);
		}
		else
		{
		    *i_minus_a_vals_it = -(*local_values_it);
		}
	    }

	    i_minus_a->insertLocalValues(
		local_row, local_cols, i_minus_a_vals() );
	}

	i_minus_a->fillComplete();
	return i_minus_a;
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_TPETRACRSMATRIXADAPTER_HPP

//---------------------------------------------------------------------------//
// end MCLS_TpetraCrsMatrixAdapter.hpp
//---------------------------------------------------------------------------//