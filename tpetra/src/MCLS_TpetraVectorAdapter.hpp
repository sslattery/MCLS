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
 * \file MCLS_TpetraVectorAdapter.hpp
 * \author Stuart R. Slattery
 * \brief Tpetra::Vector Adapter.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_TPETRAVECTORADAPTER_HPP
#define MCLS_TPETRAVECTORADAPTER_HPP

#include <MCLS_DBC.hpp>
#include <MCLS_VectorTraits.hpp>

#include <Teuchos_as.hpp>

#include <Tpetra_Vector.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class VectorTraits
 * \brief Traits specialization for Tpetra::Vector.
 */
template<class Scalar, class LO, class GO>
class VectorTraits<Tpetra::Vector<Scalar,LO,GO> >
{
  public:

    //@{
    //! Typedefs.
    typedef typename Tpetra::Vector<Scalar,LO,GO>        vector_type;
    typedef typename vector_type::scalar_type            scalar_type;
    typedef typename vector_type::local_ordinal_type     local_ordinal_type;
    typedef typename vector_type::global_ordinal_type    global_ordinal_type;
    //@}

    /*!
     * \brief Create a reference-counted pointer to a new empty vector with
     * the same parallel distribution as the given vector.
     */
    static Teuchos::RCP<vector_type> clone( const vector_type& vector )
    {
	return Tpetra::createVector<Scalar,LO,GO>( vector.getMap() );
    }

    /*!
     * \brief Create a deep copy of the provided vector and return a
     * reference-counted pointer.
     */
    static Teuchos::RCP<vector_type> deepCopy( const vector_type& vector )
    {
	return Teuchos::rcp( new Tpetra::Vector<Scalar,LO,GO>( vector ) );
    }

    /*!
     * \brief Get the global length of the vector.
     */
    static global_ordinal_type getGlobalLength( const vector_type& vector )
    { 
	return Teuchos::as<global_ordinal_type>( vector.getGlobalLength() );
    }

    /*!
     * \brief Get the local length of the vector.
     */
    static local_ordinal_type getLocalLength( const vector_type& vector )
    { 
	return Teuchos::as<local_ordinal_type>( vector.getLocalLength() );
    }

    /*!
     * \brief Determine whether or not a given global row is on-process.
     */
    static bool isGlobalRow( const vector_type& vector,
			     const global_ordinal_type& global_row )
    {
	return vector.getMap()->isNodeGlobalElement( global_row );
    }

    /*!
     * \brief Determine whether or not a given local row is on-process.
     */
    static bool isLocalRow( const vector_type& vector,
			    const local_ordinal_type& local_row )
    {
	return vector.getMap()->isNodeLocalElement( local_row );
    }

    /*!
     * \brief Given a local row on-process, provide the global ordinal.
     */
    static global_ordinal_type getGlobalRow( const vector_type& vector, 
					     const local_ordinal_type& local_row )
    { 
	return vector.getMap()->getGlobalElement( local_row );
    }

    /*!
     * \brief Given a global row on-process, provide the local ordinal.
     */
    static local_ordinal_type getLocalRow( const vector_type& vector,
					   const global_ordinal_type& global_row )
    { 
	return vector.getMap()->getLocalElement( global_row );
    }

    /*!
     * \brief Replace value at the global row index. The global index must
     * exist on process.
     */
    static void replaceGlobalValue( vector_type& vector,
    				    global_ordinal_type global_row,
    				    const scalar_type& value )
    {
	Require( vector.getMap()->isNodeGlobalElement( global_row ) );
	vector.replaceGlobalValue( global_row, value );
    }

    /*!
     * \brief Replace value at the local row index. The local index must exist
     * on process.
     */
    static void replaceLocalValue( vector_type& vector,
    				   local_ordinal_type local_row,
    				   const scalar_type& value )
    {
	Require( vector.getMap()->isNodeLocalElement( local_row ) );
	vector.replaceLocalValue( local_row, value );
    }

    /*!
     * \brief Sum a value into existing value at the global row index. The
     * global index must exist on process.
     */
    static void sumIntoGlobalValue( vector_type& vector,
    				    global_ordinal_type global_row,
    				    const scalar_type& value )
    {
	Require( vector.getMap()->isNodeGlobalElement( global_row ) );
	vector.sumIntoGlobalValue( global_row, value );
    }

    /*!
     * \brief Sum a value into existing value at the local row index. The
     * local index must exist on process.
     */
    static void sumIntoLocalValue( vector_type& vector,
    				   local_ordinal_type local_row,
    				   const scalar_type& value )
    {
	Require( vector.getMap()->isNodeLocalElement( local_row ) );
	vector.sumIntoLocalValue( local_row, value );
    }

    /*!
     * \brief Set all values in the vector to a given value.
     */
    static void putScalar( vector_type& vector, const scalar_type& value )
    { 
	vector.putScalar( value );
    }

    /*!
     * \brief Get a const view of the local vector data.
     */
    static Teuchos::ArrayRCP<const scalar_type> view( const vector_type& vector )
    { 
	return vector.getData();
    }

    /*!
     * \brief Get a non-const view of the local vector data.
     */
    static Teuchos::ArrayRCP<scalar_type> viewNonConst( vector_type& vector )
    { 
	return vector.getDataNonConst();
    }

    /*!
     * \brief Compute the dot product of two vectors A \dot B.
     */
    static scalar_type dot( const vector_type& A, const vector_type& B )
    { 
	return B.dot( A );
    }

    /*! 
     * \brief Compute the 1-norm of a vector.
     */
    static typename Teuchos::ScalarTraits<scalar_type>::magnitudeType 
    norm1( const vector_type& vector )
    {
	return vector.norm1();
    }

    /*! 
     * \brief Compute the 2-norm of a vector.
     */
    static typename Teuchos::ScalarTraits<scalar_type>::magnitudeType 
    norm2( const vector_type& vector )
    {
	return vector.norm2();
    }

    /*! 
     * \brief Compute the Inf-norm of a vector.
     */
    static typename Teuchos::ScalarTraits<scalar_type>::magnitudeType 
    normInf( const vector_type& vector )
    {
	return vector.normInf();
    }

    /*!
     * \brief Compute the mean value of a vector.
     */
    static scalar_type meanValue( const vector_type& vector )
    {
	return vector.meanValue();
    }

    /*!
     * \brief Replace output vector values with element-wise absolute values
     * of input vector A = abs(B).
     */
    static void abs( vector_type& A, const vector_type& B )
    {
	A.abs( B );
    }

    /*!
     * \brief Scale a vector by a value A = value*A.
     */
    static void scale( vector_type& A, const scalar_type value )
    {
	A.scale( value );
    }

    /*!
     * \brief Scale a vector by a value A = value*B.
     */
    static void 
    scaleCopy( vector_type& A, const scalar_type value, const vector_type& B )
    {
	A.scale( value, B );
    }
    
    /*!
     * \brief Replace output vector values with element-wise reciprocal values
     * of input vector A(i) = 1 / B(i)
     */
    static void reciprocal( vector_type& A, const vector_type& B )
    { 
	A.reciprocal( B );
    }

    /*!
     * \brief Update vector with A = alpha*A + beta*B.
     */
    static void update( vector_type& A, const scalar_type& alpha,
    			const vector_type& B, const scalar_type& beta )
    {
	A.update( beta, B, alpha );
    }

    /*!
     * \brief Update vector with A = alpha*A + beta*B + gamma*C.
     */
    static void update( vector_type& A, const scalar_type& alpha,
    			const vector_type& B, const scalar_type& beta,
    			const vector_type& C, const scalar_type& gamma )
    {
	A.update( beta, B, gamma, C, alpha );
    }

    /*!
     * \brief Element-wise mulitply two vectors 
     * A(i) = alpha*A(i) + beta*B(i)*C(i).
     */
    static void elementWiseMultiply( vector_type& A, const scalar_type& alpha,
				     const vector_type& B, const vector_type& C, 
				     const scalar_type& beta)
    { 
	A.elementWiseMultiply( beta, B, C, alpha );
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_TPETRAVECTORADAPTER_HPP

//---------------------------------------------------------------------------//
// end MCLS_TpetraVectorAdapter.hpp
//---------------------------------------------------------------------------//
