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

#include <Epetra_Map.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Import.h>

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
     * \brief Given a source matrix and an importer, build a new matrix with
     * the new decomposition.
     */
    static Teuchos::RCP<Matrix> 
    importAndFillCompleteMatrix( const Matrix& matrix, 
				 const Epetra_Import& importer )
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
	typename Teuchos::ArrayView<const int>::const_iterator global_col_it;
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

    // /*!
    //  * \brief Given a source matrix and an importer, build a new matrix with
    //  * the new decomposition.
    //  */
    // static Teuchos::RCP<matrix_type> 
    // importAndFillCompleteMatrix( const matrix_type& matrix, 
    // 				 const Epetra_Import& importer )
    // {
    // 	Require( matrix.Filled() );

    // 	Teuchos::RCP<Epetra_CrsMatrix> new_matrix = Teuchos::rcp(
    // 	    new Epetra_CrsMatrix( Copy, importer.TargetMap(), 0 ) );

    // 	new_matrix->Import( matrix, importer, Insert );
    // 	new_matrix->FillComplete();

    // 	Ensure( !new_matrix.is_null() );
    // 	return new_matrix;
    // }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_EPETRAHELPERS_HPP

//---------------------------------------------------------------------------//
// end MCLS_EpetraHelpers.hpp
//---------------------------------------------------------------------------//
