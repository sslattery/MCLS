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
 * \file MCLS_OperatorTraits.hpp
 * \author Stuart R. Slattery
 * \brief Operator traits definition.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_OPERATORTRAITS_HPP
#define MCLS_OPERATORTRAITS_HPP

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ArrayView.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class UndefinedOperatorTraits
 * \brief Class for undefined operator traits. 
 *
 * Will throw a compile-time error if these traits are not specialized.
 */
template<class OperatorType>
struct UndefinedOperatorTraits
{
    static inline OperatorType notDefined()
    {
	return OperatorType::this_type_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class OperatorTraits
 * \brief Traits for matrix operators.
 *
 * OperatorTraits defines an interface for parallel distributed matrix
 * operators.  (e.g. Tpetra::CrsMatrix or Epetra_VbrMatrix).
 */
template<class OperatorType>
class OperatorTraits
{
  public:

    //@{
    //! Typedefs.
    typedef OperatorType                                  operator_type;
    typedef VectorType                                    vector_type;
    typedef typename operator_type::scalar_type           scalar_type;
    typedef typename operator_type::local_ordinal_type    local_ordinal_type;
    typedef typename operator_type::global_ordinal_type   global_ordinal_type;
    //@}

    /*!
     * \brief Create a reference-counted pointer to a new empty operator with
     * the same parallel distribution as the given operator.
     */
    static Teuchos::RCP<operator_type> 
    clone( const operator_type& op )
    { 
	UndefinedOperatorTraits<operator_type>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Create a reference-counted pointer to a new empty vector from a
     * matrix to give the vector the same parallel distribution as the
     * matrix.
     */
    static Teuchos::RCP<vector_type> 
    cloneVectorFromOperator( const operator_type& op )
    { 
	UndefinedOperatorTraits<operator_type>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Get the communicator.
     */
    static const Teuchos::RCP<const Teuchos::Comm<int> >&
    getComm( const operator_type& op )
    {
	UndefinedOperatorTraits<operator_type>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Get the global number of rows.
     */
    static global_ordinal_type getGlobalNumRows( const operator_type& op )
    { UndefinedOperatorTraits<operator_type>::notDefined(); return 0; }

    /*!
     * \brief Get the local number of rows.
     */
    static local_ordinal_type getLocalNumRows( const operator_type& op )
    { UndefinedOperatorTraits<operator_type>::notDefined(); return 0; }

    /*!
     * \brief Get the maximum number of entries in a row globally.
     */
    static global_ordinal_type 
    getGlobalMaxNumRowEntries( const operator_type& op )
    { UndefinedOperatorTraits<operator_type>::notDefined(); return 0; }

    /*!
     * \brief Given a local row on process, provide the global ordinal.
     */
    static global_ordinal_type 
    getGlobalRow( const operator_type& op,
		  const local_ordinal_type& local_ordinal )
    { UndefinedOperatorTraits<operator_type>::notDefined(); return 0; }

    /*!
     * \brief Given a global row on process, provide the local ordinal.
     */
    static local_ordinal_type 
    getLocalRow( const operator_type& op,
		 const global_ordinal_type& global_ordinal )
    { UndefinedOperatorTraits<operator_type>::notDefined(); return 0; }

    /*!
     * \brief Given a local col on process, provide the global ordinal.
     */
    static global_ordinal_type 
    getGlobalCol( const operator_type& op,
		  const local_ordinal_type& local_ordinal )
    { UndefinedOperatorTraits<operator_type>::notDefined(); return 0; }

    /*!
     * \brief Given a global col on process, provide the local ordinal.
     */
    static local_ordinal_type 
    getLocalCol( const operator_type& op,
		 const global_ordinal_type& global_ordinal )
    { UndefinedOperatorTraits<operator_type>::notDefined(); return 0; }

    /*!
     * \brief Determine whether or not a given global row is on process.
     */
    static bool isGlobalRow( const operator_type& op,
			     const global_ordinal_type& global_ordinal )
    { UndefinedOperatorTraits<operator_type>::notDefined(); return false; }

    /*!
     * \brief Determine whether or not a given local row is on process.
     */
    static bool isLocalRow( const local_ordinal_type& local_ordinal )
    { UndefinedOperatorTraits<operator_type>::notDefined(); return false; }

    /*!
     * \brief Determine whether or not a given global col is on process.
     */
    static bool isGlobalCol( const operator_type& op,
			     const global_ordinal_type& global_ordinal )
    { UndefinedOperatorTraits<operator_type>::notDefined(); return false; }

    /*!
     * \brief Determine whether or not a given local col is on process.
     */
    static bool isLocalCol( const operator_type& op,
			    const local_ordinal_type& local_ordinal )
    { UndefinedOperatorTraits<operator_type>::notDefined(); return false; }

    /*!
     * \brief Get a view of a global row.
     */
    static void getGlobalRowView( 
	const operator_type& op,
	const global_ordinal_type& global_ordinal, 
	Teuchos::ArrayView<const global_ordinal_type> &indices,
	Teuchos::ArrayView<const scalar_type> &values)
    { UndefinedOperatorTraits<operator_type>::notDefined(); }

    /*!
     * \brief Get a view of a local row.
     */
    static void getLocalRowView( 
	const operator_type& op,
	const local_ordinal_type& local_ordinal, 
	Teuchos::ArrayView<const local_ordinal_type> &indices,
	Teuchos::ArrayView<const scalar_type> &values)
    { UndefinedOperatorTraits<operator_type>::notDefined(); }

    /*!
     * \brief Get a copy of the local diagonal of the matrix.
     */
    static void getLocalDiagCopy( const operator_type& op, vector_type& vector )
    { 
	UndefinedOperatorTraits<operator_type>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Apply the row matrix to a vector. A*x = y.
     */
    static void apply( const operator_type& A, const vector_type& x,
		       const vector_type& y )
    { UndefinedOperatorTraits<operator_type>::notDefined(); }

    /*
     * \brief Create a reference-counted pointer to a new matrix with a
     * specified number of off process nearest-neighbor global rows.
     */
    static Teuchos::RCP<operator_type> copyNearestNeighbors(
	const operator_type& op, const global_ordinal_type& num_neighbors )
    { 
	UndefinedOperatorTraits<operator_type>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Create a reference-counted pointer to a new matrix by
     * subtracting the transpose of a matrix from the identity matrix.
     */
    static Teuchos::RCP<operator_type> 
    subtractTransposeFromIdentity( const operator_type& op )
    { 
	UndefinedOperatorTraits<operator_type>::notDefined(); 
	return Teuchos::null; 
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_OPERATORTRAITS_HPP

//---------------------------------------------------------------------------//
// end MCLS_OperatorTraits.hpp
// ---------------------------------------------------------------------------//

