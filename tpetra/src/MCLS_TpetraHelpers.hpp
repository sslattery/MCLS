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
 * \file MCLS_TpetraAdapater.hpp
 * \author Stuart R. Slattery
 * \brief Tpetra Helpers.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_TPETRAHELPERS_HPP
#define MCLS_TPETRAHELPERS_HPP

#include <MCLS_DBC.hpp>

#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Import.hpp>

//---------------------------------------------------------------------------//
/*!
 * \class UndefinedTpetraHelpers
 * \brief Class for undefined TpetraHelper functions.
 *
 * Will throw a compile-time error if these traits are not specialized.
 */
template<class Scalar, class LO, class GO, class Matrix>
struct UndefinedTpetraHelpers
{
    static inline void notDefined()
    {
	return Matrix::this_type_is_missing_a_specialization();
    }
};

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class TpetraMatrixHelpers
 * \brief Helper functions for Tpetra implementations.
 */
template<class Scalar, class LO, class GO, class Matrix>
class TpetraMatrixHelpers
{
  public:

    //@{
    //! Typedefs.
    typedef Scalar                                  scalar_type;
    typedef LO                                      local_ordinal_type;
    typedef GO                                      global_ordinal_type;
    typedef Matrix                                  matrix_type;
    //@}

    /*!
     * \brief Get the on-process global matrix column indices that, as global
     * row indices, are off-process.
     */
    static Teuchos::Array<GO> getOffProcColsAsRows( const Matrix& matrix )
    { 
	UndefinedTpetraHelpers<Scalar,LO,GO,Matrix>::notDefined(); 
	return Teuchos::Array<GO>(0); 
    }

    /*!
     * \brief Given a source matrix and an importer, build a new matrix with
     * the new decomposition.
     */
    static Teuchos::RCP<Matrix> 
    importAndFillCompleteMatrix( const Matrix& matrix, 
				 const Tpetra::Import<LO,GO>& importer )
    {
	UndefinedTpetraHelpers<Scalar,LO,GO,Matrix>::notDefined(); 
	return Teuchos::null;
    }

    /*!
     * \brief Filter values out of a matrix below a certain threshold.
     */
    static Teuchos::RCP<Matrix> filter(
    	const Matrix& matrix, const double& threshold )
    {
	UndefinedTpetraHelpers<Scalar,LO,GO,Matrix>::notDefined(); 
	return Teuchos::null;
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class TpetraMatrixHelpers
 * \brief TpetraMatrixHelpers specialization for Tpetra::CrsMatrix.
 */
template<class Scalar, class LO, class GO>
class TpetraMatrixHelpers<Scalar,LO,GO,Tpetra::CrsMatrix<Scalar,LO,GO> >
{
  public:

    //@{
    //! Typedefs.
    typedef Scalar                                        scalar_type;
    typedef LO                                            local_ordinal_type;
    typedef GO                                            global_ordinal_type;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO>               matrix_type;
    //@}

    /*!
     * \brief Get the on-process global matrix column indices that, as global
     * row indices, are off-process.
     */
    static Teuchos::Array<GO> getOffProcColsAsRows( const matrix_type& matrix )
    { 
	MCLS_REQUIRE( matrix.isFillComplete() );

	Teuchos::RCP<const Tpetra::Map<LO,GO> > row_map = 
	    matrix.getRowMap();
	Teuchos::RCP<const Tpetra::Map<LO,GO> > col_map = 
	    matrix.getColMap();

	Teuchos::ArrayView<const GO> global_cols = 
	    col_map->getNodeElementList();

	Teuchos::Array<GO> off_proc_cols(0);
	typename Teuchos::ArrayView<const GO>::const_iterator global_col_it;
	for ( global_col_it = global_cols.begin();
	      global_col_it != global_cols.end();
	      ++global_col_it )
	{
	    if ( !row_map->isNodeGlobalElement( *global_col_it ) )
	    {
		off_proc_cols.push_back( *global_col_it );
	    }
	}

	return off_proc_cols;
    }

    /*!
     * \brief Given a source matrix and an importer, build a new matrix with
     * the new decomposition.
     */
    static Teuchos::RCP<matrix_type> 
    importAndFillCompleteMatrix( const matrix_type& matrix, 
				 const Tpetra::Import<LO,GO>& importer )
    {
	MCLS_REQUIRE( matrix.isFillComplete() );

	Teuchos::RCP<matrix_type> new_matrix = Teuchos::rcp(
	    new matrix_type( importer.getTargetMap(), 0 ) );

	new_matrix->doImport( matrix, importer, Tpetra::INSERT );
	new_matrix->fillComplete( matrix.getDomainMap(), matrix.getRangeMap() );

	MCLS_ENSURE( !new_matrix.is_null() );
	return new_matrix;
    }

    /*!
     * \brief Filter values out of a matrix below a certain threshold. The
     * threshold is relative to the maximum value in each row of the matrix.
     */
    static Teuchos::RCP<matrix_type> filter(
    	const matrix_type& A, const double& threshold )
    {
	const Teuchos::RCP<const Tpetra::Map<LO,GO> > row_map = 
	    A.getRowMap(); 
	const Teuchos::RCP<const Tpetra::Map<LO,GO> > col_map = 
	    A.getColMap(); 
    
        Teuchos::RCP<matrix_type> A_filter = Teuchos::rcp(
            new matrix_type(row_map, 0) );

        LO num_local_rows = row_map->getNodeNumElements();

        LO max_entries = A.getGlobalMaxNumRowEntries();
	std::size_t num_entries = 0;
        Teuchos::Array<LO> local_indices(max_entries);
        Teuchos::Array<GO> global_indices(max_entries);
        Teuchos::Array<Scalar> values(max_entries);
	typename Teuchos::Array<GO>::iterator index_it;
	typename Teuchos::Array<Scalar>::iterator value_it;
	Scalar row_threshold = 0.0;

	// Process row-by-row.
        for( int local_row = 0; local_row < num_local_rows; ++local_row ) 
        {
	    // Get a copy of the row.
	    A.getLocalRowCopy( 
		local_row, local_indices(), values(), num_entries );

	    // Get the threshold for this row.
	    row_threshold = 
		*std::max_element(values.getRawPtr(),
				  values.getRawPtr()+num_entries)*threshold;

	    // Find values below the threshold.
            for (int j = 0 ; j < num_entries; ++j ) 
            { 
		if ( std::abs(values[j]) <= row_threshold )
		{
		    values[j] = 0.0;
		    global_indices[j] = -1;
		}
		else
		{
		    global_indices[j] = 
			col_map->getGlobalElement( local_indices[j] );
		}
            }

	    // Remove values below the threshold.
	    value_it = 
		std::remove( values.begin(), values.begin()+num_entries, 0.0 );
	    index_it = std::remove( 
		global_indices.begin(), global_indices.begin()+num_entries, -1 );
	    num_entries = std::distance( values.begin(), value_it );
	    MCLS_CHECK( std::distance(global_indices.begin(), index_it) == 
			num_entries );

	    // Create a new row in the filtered matrix.
	    A_filter->insertGlobalValues( row_map->getGlobalElement(local_row), 
					  global_indices(0,num_entries),
					  values(0,num_entries) );
        }

	A_filter->fillComplete();

        return A_filter;
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_TPETRAHELPERS_HPP

//---------------------------------------------------------------------------//
// end MCLS_TpetraHelpers.hpp
//---------------------------------------------------------------------------//
