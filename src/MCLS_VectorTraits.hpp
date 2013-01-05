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
template<class Scalar, class LO, class GO, class Vector>
struct UndefinedVectorTraits
{
    static inline void notDefined()
    {
	return Vector::this_type_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class VectorTraits
 * \brief Traits for vectors.
 *
 * VectorTraits defines an interface for parallel distributed vectors
 * (e.g. Tpetra::Vector or Epetra_Vector).
 */
template<class Scalar, class LO, class GO, class Vector>
class VectorTraits
{
  public:

    //@{
    //! Typedefs.
    typedef Scalar                                  scalar_type;
    typedef LO                                      local_ordinal_type;
    typedef GO                                      global_ordinal_type;
    typedef Vector                                  vector_type;
    //@}

    /*!
     * \brief Create a reference-counted pointer to a new empty vector with
     * the same parallel distribution as the given vector.
     */
    static Teuchos::RCP<Vector> clone( const Vector& vector )
    { 
	UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Create a deep copy of the provided vector and return a
     * reference-counted pointer.
     */
    static Teuchos::RCP<Vector> deepCopy( const Vector& vector )
    { 
	UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Get the global length of the vector.
     */
    static GO getGlobalLength( const Vector& vector )
    { 
	UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); 
	return 0; 
    }

    /*!
     * \brief Get the local length of the vector.
     */
    static LO getLocalLength( const Vector& vector )
    { 
	UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); 
	return 0; 
    }

    /*!
     * \brief Replace value at the global row index. The global index must
     * exist on process.
     */
    static void replaceGlobalValue( Vector& vector, 
				    GO global_row,  
				    const Scalar& value )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); }

    /*!
     * \brief Replace value at the local row index. The local index must exist
     * on process.
     */
    static void replaceLocalValue( Vector& vector,
				   LO local_row,
				   const Scalar& value )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); }

    /*!
     * \brief Sum a value into existing value at the global row index. The
     * global index must exist on process.
     */
    static void sumIntoGlobalValue( Vector& vector,
				    GO global_row,
				    const Scalar& value )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); }

    /*!
     * \brief Sum a value into existing value at the local row index. The
     * local index must exist on process.
     */
    static void sumIntoLocalValue( Vector& vector,
				   LO local_row,
				   const Scalar& value )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); }

    /*!
     * \brief Set all values in the vector to a given value.
     */
    static void putScalar( Vector& vector, const Scalar& value )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); }

    /*!
     * \brief Get a const view of the local vector data.
     */
    static Teuchos::ArrayRCP<const Scalar> view( const Vector& vector )
    { 
	UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); 
	return Teuchos::ArrayRCP<const Scalar>(0,0);
    }

    /*!
     * \brief Get a non-const view of the local vector data.
     */
    static Teuchos::ArrayRCP<Scalar> viewNonConst( Vector& vector )
    { 
	UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); 
	return Teuchos::ArrayRCP<Scalar>(0,0);
    }

    /*!
     * \brief Compute the dot product of two vectors A \dot B.
     */
    static Scalar dot( const Vector& A, const Vector& B )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); return 0; }

    /*! 
     * \brief Compute the 1-norm of a vector.
     */
    static typename Teuchos::ScalarTraits<Scalar>::magnitudeType 
    norm1( const Vector& vector )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); return 0; }

    /*! 
     * \brief Compute the 2-norm of a vector.
     */
    static typename Teuchos::ScalarTraits<Scalar>::magnitudeType 
    norm2( const Vector& vector )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); return 0; }

    /*! 
     * \brief Compute the Inf-norm of a vector.
     */
    static typename Teuchos::ScalarTraits<Scalar>::magnitudeType 
    normInf( const Vector& vector )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); return 0; }

    /*!
     * \brief Compute the mean value of a vector.
     */
    static Scalar meanValue( const Vector& vector )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); return 0; }

    /*!
     * \brief Replace output vector values with element-wise absolute values
     * of input vector A = abs(B).
     */
    static void abs( Vector& A, const Vector& B )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); }

    /*!
     * \brief Scale a vector by a value A = value*A.
     */
    static void scale( Vector& A, const Scalar value )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); }

    /*!
     * \brief Scale a vector by a value A = value*B.
     */
    static void scaleCopy( Vector& A, const Scalar value, const Vector& B )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); }
    
    /*!
     * \brief Replace output vector values with element-wise reciprocal values
     * of input vector A(i) = 1 / B(i)
     */
    static void reciprocal( Vector& A, const Vector& B )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); }

    /*!
     * \brief Update vector with A = alpha*A + beta*B.
     */
    static void update( Vector& A, const Scalar& alpha,
			const Vector& B, const Scalar& beta )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); }

    /*!
     * \brief Update vector with A = alpha*A + beta*B + gamma*C.
     */
    static void update( Vector& A, const Scalar& alpha,
			const Vector& B, const Scalar& beta,
			const Vector& C, const Scalar& gamma )
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); }

    /*!
     * \brief Element-wise mulitply two vectors 
     * A(i) = alpha*A(i) + beta*B(i)*C(i).
     */
    static void elementWiseMultiply( Vector& A, const Scalar& alpha,
				     const Vector& B, const Vector& C,
				     const Scalar& beta)
    { UndefinedVectorTraits<Scalar,LO,GO,Vector>::notDefined(); }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_VECTORTRAITS_HPP

//---------------------------------------------------------------------------//
// end MCLS_VectorTraits.hpp
//---------------------------------------------------------------------------//

