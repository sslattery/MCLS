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
 * \file MCLS_VectorTraits.hpp
 * \author Stuart R. Slattery
 * \brief Vector traits definition.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_VECTORTRAITS_HPP
#define MCLS_VECTORTRAITS_HPP

#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_ScalarTraits.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class UndefinedVectorTraits
 * \brief Class for undefined vector traits. 
 *
 * Will throw a compile-time error if these traits are not specialized.
 */
template<class VectorType>
struct UndefinedVectorTraits
{
    static inline VectorType notDefined()
    {
	return VectorType::this_type_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class VectorTraits
 * \brief Traits for vectors.
 *
 * VectorTraits defines an interface for parallel distributed vectors
 * (e.g. Tpetra::Vector or Epetra_Multivector).
 */
template<class VectorType>
class VectorTraits
{
  public:

    //@{
    //! Typedefs.
    typedef VectorType                                   vector_type;
    typedef typename vector_type::scalar_type            scalar_type;
    typedef typename vector_type::local_ordinal_type     local_ordinal_type;
    typedef typename vector_type::global_ordinal_type    global_ordinal_type;
    //@}

    /*!
     * \brief Create a reference-counted pointer to a new empty vector with
     * the same parallel distribution as the given vector.
     */
    static Teuchos::RCP<vector_type> clone( const vector_type& vector )
    { UndefinedVectorTraits<vector_type>::notDefined(); return Teuchos::null; }

    /*!
     * \brief Create a deep copy of the provided vector and return a
     * reference-counted pointer.
     */
    static Teuchos::RCP<vector_type> deepCopy( const vector_type& vector )
    { UndefinedVectorTraits<vector_type>::notDefined(); return Teuchos::null; }

    /*!
     * \brief Get the global length of the vector.
     */
    static global_ordinal_type getGlobalLength( const vector_type& vector )
    { UndefinedVectorTraits<vector_type>::notDefined(); return 0; }

    /*!
     * \brief Get the local length of the vector.
     */
    static local_ordinal_type getLocalLength( const vector_type& vector )
    { UndefinedVectorTraits<vector_type>::notDefined(); return 0; }

    /*!
     * \brief Replace value at the global row index. The global index must
     * exist on process.
     */
    static void replaceGlobalValue( vector_type& vector,
				    global_ordinal_type global_row,
				    const scalar_type& value )
    { UndefinedVectorTraits<vector_type>::notDefined(); }

    /*!
     * \brief Replace value at the local row index. The local index must exist
     * on process.
     */
    static void replaceLocalValue( vector_type& vector,
				   local_ordinal_type local_row,
				   const scalar_type& value )
    { UndefinedVectorTraits<vector_type>::notDefined(); }

    /*!
     * \brief Sum a value into existing value at the global row index. The
     * global index must exist on process.
     */
    static void sumIntoGlobalValue( vector_type& vector,
				    global_ordinal_type global_row,
				    const scalar_type& value )
    { UndefinedVectorTraits<vector_type>::notDefined(); }

    /*!
     * \brief Sum a value into existing value at the local row index. The
     * local index must exist on process.
     */
    static void sumIntoLocalValue( vector_type& vector,
				   local_ordinal_type local_row,
				   const scalar_type& value )
    { UndefinedVectorTraits<vector_type>::notDefined(); }

    /*!
     * \brief Set all values in the vector to a given value.
     */
    static void putScalar( vector_type& vector, const scalar_type& value )
    { UndefinedVectorTraits<vector_type>::notDefined(); }

    /*!
     * \brief Get a const view of the local vector data.
     */
    static Teuchos::ArrayRCP<const scalar_type> view( const vector_type& vector )
    { 
	UndefinedVectorTraits<vector_type>::notDefined(); 
	return Teuchos::ArrayRCP<const scalar_type>(0,0);
    }

    /*!
     * \brief Get a non-const view of the local vector data.
     */
    static Teuchos::ArrayRCP<scalar_type> 
    viewNonConst( vector_type& vector )
    { 
	UndefinedVectorTraits<vector_type>::notDefined(); 
	return Teuchos::ArrayRCP<scalar_type>(0,0);
    }

    /*!
     * \brief Compute the dot product of two vectors A \dot B.
     */
    static scalar_type 
    dot( const vector_type& A, const vector_type& B )
    { UndefinedVectorTraits<vector_type>::notDefined(); return 0; }

    /*! 
     * \brief Compute the 1-norm of a vector.
     */
    static typename Teuchos::ScalarTraits<scalar_type>::magnitudeType 
    norm1( const vector_type& vector )
    { UndefinedVectorTraits<vector_type>::notDefined(); return 0; }

    /*! 
     * \brief Compute the 2-norm of a vector.
     */
    static typename Teuchos::ScalarTraits<scalar_type>::magnitudeType 
    norm2( const vector_type& vector )
    { UndefinedVectorTraits<vector_type>::notDefined(); return 0; }

    /*! 
     * \brief Compute the Inf-norm of a vector.
     */
    static typename Teuchos::ScalarTraits<scalar_type>::magnitudeType 
    normInf( const vector_type& vector )
    { UndefinedVectorTraits<vector_type>::notDefined(); return 0; }

    /*!
     * \brief Compute the mean value of a vector.
     */
    static scalar_type meanValue( const vector_type& vector )
    { UndefinedVectorTraits<vector_type>::notDefined(); return 0; }

    /*!
     * \brief Replace output vector values with element-wise absolute values
     * of input vector A = abs(B).
     */
    static void abs( vector_type& A, const vector_type& B )
    { UndefinedVectorTraits<vector_type>::notDefined(); }

    /*!
     * \brief Scale a vector by a value A = value*A.
     */
    static void scale( vector_type& A, const scalar_type value )
    { UndefinedVectorTraits<vector_type>::notDefined(); }

    /*!
     * \brief Scale a vector by a value A = value*B.
     */
    static void 
    scaleCopy( vector_type& A, const scalar_type value, const vector_type& B )
    { UndefinedVectorTraits<vector_type>::notDefined(); }
    
    /*!
     * \brief Replace output vector values with element-wise reciprocal values
     * of input vector A(i) = 1 / B(i)
     */
    static void reciprocal( vector_type& A, const vector_type& B )
    { UndefinedVectorTraits<vector_type>::notDefined(); }

    /*!
     * \brief Update vector with A = alpha*A + beta*B.
     */
    static void update( vector_type& A, const scalar_type& alpha,
			const vector_type& B, const scalar_type& beta )
    { UndefinedVectorTraits<vector_type>::notDefined(); }

    /*!
     * \brief Update vector with A = alpha*A + beta*B + gamma*C.
     */
    static void update( vector_type& A, const scalar_type& alpha,
			const vector_type& B, const scalar_type& beta,
			const vector_type& C, const scalar_type& gamma )
    { UndefinedVectorTraits<vector_type>::notDefined(); }

    /*!
     * \brief Element-wise mulitply two vectors 
     * A(i) = alpha*A(i) + beta*B(i)*C(i).
     */
    static void elementWiseMultiply( vector_type& A, const scalar_type& alpha,
				     const vector_type& B, const vector_type& C,
				     const scalar_type& beta)
    { UndefinedVectorTraits<vector_type>::notDefined(); }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_VECTORTRAITS_HPP

//---------------------------------------------------------------------------//
// end MCLS_VectorTraits.hpp
//---------------------------------------------------------------------------//

