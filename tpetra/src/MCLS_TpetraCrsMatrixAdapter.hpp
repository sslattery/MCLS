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
    static GO getGlobalRow( const matrix_type& matrix, const LO& local_ordinal )
    { 
	return matrix.getRowMap()->getGlobalElement( local_ordinal );
    }

    /*!
     * \brief Given a global row on-process, provide the local ordinal.
     */
    static LO getLocalRow( const matrix_type& matrix, const GO& global_ordinal )
    { 
	return matrix.getRowMap()->getLocalElement( global_ordinal );
    }

    /*!
     * \brief Given a local col on-process, provide the global ordinal.
     */
    static GO getGlobalCol( const matrix_type& matrix, const LO& local_ordinal )
    {
	testPrecondition( matrix.isFillComplete() );
	return matrix.getColMap()->getGlobalElement( local_ordinal );
    }

    /*!
     * \brief Given a global col on-process, provide the local ordinal.
     */
    static LO getLocalCol( const matrix_type& matrix, const GO& global_ordinal )
    {
	testPrecondition( matrix.isFillComplete() );
	return matrix.getColMap()->getLocalElement( global_ordinal );
    }

    /*!
     * \brief Determine whether or not a given global row is on-process.
     */
    static bool isGlobalRow( const matrix_type& matrix, const GO& global_ordinal )
    {
	return matrix.getRowMap()->isNodeGlobalElement( global_ordinal );
    }

    /*!
     * \brief Determine whether or not a given local row is on-process.
     */
    static bool isLocalRow( const matrix_type& matrix, const LO& local_ordinal )
    { 
	return matrix.getRowMap()->isNodeLocalElement( local_ordinal );
    }

    /*!
     * \brief Determine whether or not a given global col is on-process.
     */
    static bool isGlobalCol( const matrix_type& matrix, const GO& global_ordinal )
    { 
	testPrecondition( matrix.isFillComplete() );
	return matrix.getColMap()->isNodeGlobalElement( global_ordinal );
    }

    /*!
     * \brief Determine whether or not a given local col is on-process.
     */
    static bool isLocalCol( const matrix_type& matrix, const LO& local_ordinal )
    { 
	testPrecondition( matrix.isFillComplete() );
	return matrix.getColMap()->isNodeLocalElement( local_ordinal );
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_TPETRACRSMATRIXADAPTER_HPP

//---------------------------------------------------------------------------//
// end MCLS_TpetraCrsMatrixAdapter.hpp
//---------------------------------------------------------------------------//
