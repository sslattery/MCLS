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
 * \file MCLS_EpetraAdapater.hpp
 * \author Stuart R. Slattery
 * \brief Epetra Helpers.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_EPETRAHELPERS_HPP
#define MCLS_EPETRAHELPERS_HPP

#include <MCLS_DBC.hpp>

#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_as.hpp>

#include <Epetra_Map.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Export.h>
#include <Epetra_RowMatrixTransposer.h>

//---------------------------------------------------------------------------//
/*!
 * \class UndefinedEpetraHelpers
 * \brief Class for undefined EpetraHelper functions.
 *
 * Will throw a compile-time error if these traits are not specialized.
 */
template<class Matrix>
struct UndefinedEpetraHelpers
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
 * \class EpetraMatrixHelpers
 * \brief Helper functions for Epetra implementations.
 */
template<class Matrix>
class EpetraMatrixHelpers
{
  public:

    //@{
    //! Typedefs.
    typedef Matrix                                  matrix_type;
    //@}

    /*!
     * \brief Get the on-process global matrix column indices that, as global
     * row indices, are off-process.
     */
    static Teuchos::Array<int> getOffProcColsAsRows( const Matrix& matrix )
    { 
	UndefinedEpetraHelpers<Matrix>::notDefined(); 
	return Teuchos::Array<int>(0); 
    }

    /*!
     * \brief Get a copy of the transpose of a matrix.
     */
    static Teuchos::RCP<matrix_type> copyTranspose( const matrix_type& matrix )
    { 
	UndefinedEpetraHelpers<Matrix>::notDefined(); 
	return Teuchos::null;
    }

    /*
     * \brief Create a reference-counted pointer to a new matrix with a
     * specified number of off-process nearest-neighbor global rows.
     */
    static Teuchos::RCP<matrix_type> copyNearestNeighbors( 
    	const matrix_type& matrix, const int& num_neighbors )
    { 
	UndefinedEpetraHelpers<Matrix>::notDefined(); 
	return Teuchos::null;
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class EpetraMatrixHelpers
 * \brief EpetraMatrixHelpers specialization for Epetra_RowMatrix.
 */
template<>
class EpetraMatrixHelpers<Epetra_RowMatrix>
{
  public:

    //@{
    //! Typedefs.
    typedef Epetra_RowMatrix               matrix_type;
    //@}

    /*!
     * \brief Get the on-process global matrix column indices that, as global
     * row indices, are off-process.
     */
    static Teuchos::Array<int> getOffProcColsAsRows( const matrix_type& matrix )
    { 
	Require( matrix.Filled() );

	const Epetra_Map& row_map = matrix.RowMatrixRowMap();
	const Epetra_Map& col_map = matrix.RowMatrixColMap();

	Teuchos::ArrayView<const int> global_cols( col_map.MyGlobalElements(),
						   col_map.NumMyElements() );

	Teuchos::Array<int> off_proc_cols(0);
	Teuchos::ArrayView<const int>::const_iterator global_col_it;
	for ( global_col_it = global_cols.begin();
	      global_col_it != global_cols.end();
	      ++global_col_it )
	{
	    if ( !row_map.MyGID( *global_col_it ) )
	    {
		off_proc_cols.push_back( *global_col_it );
	    }
	}

	return off_proc_cols;
    }

    /*!
     * \brief Get a copy of the transpose of a matrix.
     */
    static Teuchos::RCP<matrix_type> copyTranspose( const matrix_type& matrix )
    { 
	Epetra_RowMatrixTransposer transposer( const_cast<matrix_type*>(&matrix) );

	Epetra_CrsMatrix* transpose_matrix;
	transposer.CreateTranspose( true, transpose_matrix );

	Ensure( transpose_matrix->Filled() );
	return Teuchos::RCP<matrix_type>( transpose_matrix );
    }

    /*
     * \brief Create a reference-counted pointer to a new matrix with a
     * specified number of off-process nearest-neighbor global rows.
     */
    static Teuchos::RCP<matrix_type> copyNearestNeighbors( 
    	const matrix_type& matrix, const int& num_neighbors )
    { 
	Require( num_neighbors >= 0 ); 

	// Setup for neighbor construction.
	Teuchos::RCP<const Epetra_Map> empty_map = Teuchos::rcp(
	    new Epetra_Map( 0, 0, matrix.Comm() ) );
	Teuchos::RCP<Epetra_CrsMatrix> neighbor_matrix = 
	    Teuchos::rcp( new Epetra_CrsMatrix( Copy, *empty_map, 0 ) );
	neighbor_matrix->FillComplete();

	Teuchos::ArrayView<const int> global_rows;
	Teuchos::ArrayView<const int>::const_iterator global_rows_it;
	Teuchos::Array<int>::iterator ghost_global_bound;

	// Get the initial off proc columns.
	Teuchos::Array<int> ghost_global_rows = getOffProcColsAsRows( matrix );

	// Build the neighbors by traversing the graph.
	for ( int i = 0; i < num_neighbors; ++i )
	{
	    // Get rid of the global rows that belong to the original
	    // matrix. We don't need to store these, just the neighbors.
	    global_rows = Teuchos::ArrayView<const int>( 
		matrix.RowMatrixRowMap().MyGlobalElements(),
		matrix.RowMatrixRowMap().NumMyElements() );
	    for ( global_rows_it = global_rows.begin();
		  global_rows_it != global_rows.end();
		  ++global_rows_it )
	    {
		ghost_global_bound = std::remove( ghost_global_rows.begin(), 
						  ghost_global_rows.end(), 
						  *global_rows_it );
		ghost_global_rows.resize( std::distance(ghost_global_rows.begin(),
							ghost_global_bound) );
	    }

	    // Get the current set of global rows in the neighbor matrix. 
	    global_rows = Teuchos::ArrayView<const int>( 
		neighbor_matrix->RowMatrixRowMap().MyGlobalElements(),
		neighbor_matrix->RowMatrixRowMap().NumMyElements() );

	    // Append the on proc neighbor columns to the off proc columns.
	    for ( global_rows_it = global_rows.begin();
		  global_rows_it != global_rows.end();
		  ++global_rows_it )
	    {
		ghost_global_rows.push_back( *global_rows_it );
	    }
	
	    // Make a new map of the combined global rows and off proc columns.
	    Teuchos::RCP<const Epetra_Map> ghost_map = Teuchos::rcp( 
		new Epetra_Map( -1, 
				Teuchos::as<int>(ghost_global_rows.size()),
				ghost_global_rows.getRawPtr(),
				0,
				neighbor_matrix->Comm() ) );

	    // Export the neighbor matrix with the new neighbor.
	    Epetra_Export ghost_exporter( matrix.RowMatrixRowMap(), *ghost_map );

	    neighbor_matrix = Teuchos::rcp( 
		new Epetra_CrsMatrix( Copy, *ghost_map, 0 ) );

	    neighbor_matrix->Export( matrix, ghost_exporter, Insert );
	    neighbor_matrix->FillComplete();

	    // Get the next rows in the graph.
	    ghost_global_rows = getOffProcColsAsRows( *neighbor_matrix );
	}

	Ensure( !neighbor_matrix.is_null() );
	Ensure( neighbor_matrix->Filled() );
	return neighbor_matrix;
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_EPETRAHELPERS_HPP

//---------------------------------------------------------------------------//
// end MCLS_EpetraHelpers.hpp
//---------------------------------------------------------------------------//
