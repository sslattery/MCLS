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
 * \file MCLS_EpetraVectorAdapater.hpp
 * \author Stuart R. Slattery
 * \brief Epetra_Vector Adapter.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_EPETRAVECTORADAPTER_HPP
#define MCLS_EPETRAVECTORADAPTER_HPP

#include <MCLS_DBC.hpp>
#include <MCLS_VectorTraits.hpp>

#include <Teuchos_as.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_OpaqueWrapper.hpp>

#include <Epetra_Vector.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Map.h>
#include <Epetra_Comm.h>
#include <Epetra_SerialComm.h>

#ifdef HAVE_MPI
#include <Teuchos_DefaultMpiComm.hpp>
#include <Epetra_MpiComm.h>
#endif

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class VectorTraits
 * \brief Traits specialization for Epetra_Vector.
 */
template<>
class VectorTraits<Epetra_Vector>
{
  public:

    //@{
    //! Typedefs.
    typedef Epetra_Vector                       vector_type;
    typedef double                              scalar_type;
    typedef int                                 local_ordinal_type;
    typedef int                                 global_ordinal_type;
    typedef Epetra_MultiVector                  multivector_type;
    typedef Teuchos::Comm<int>                  Comm;
    //@}

    /*!
     * \brief Create a reference-counted pointer to a new empty vector with
     * the same parallel distribution as the given vector.
     */
    static Teuchos::RCP<vector_type> clone( const vector_type& vector )
    {
	return Teuchos::rcp( new Epetra_Vector( vector.Map() ) );
    }

    /*!
     * \brief Create a reference-counted pointer to a new empty vector with
     * the same parallel distribution given by the input rows.
     */
    static Teuchos::RCP<vector_type> 
    createFromRows( 
	const Teuchos::RCP<const Comm>& comm,
	const Teuchos::ArrayView<const global_ordinal_type>& global_rows )
    { 
	Teuchos::RCP<Epetra_Comm> epetra_comm;
#ifdef HAVE_MPI
	Teuchos::RCP< const Teuchos::MpiComm<int> > mpi_comm = 
	    Teuchos::rcp_dynamic_cast< const Teuchos::MpiComm<int> >( comm );
	Teuchos::RCP< const Teuchos::OpaqueWrapper<MPI_Comm> > opaque_comm = 
	    mpi_comm->getRawMpiComm();
	epetra_comm = Teuchos::rcp( new Epetra_MpiComm( (*opaque_comm)() ) );
#else
	epetra_comm = Teuchos::rcp( new Epetra_SerialComm() );
#endif
	Teuchos::RCP<const Epetra_Map> map = Teuchos::rcp( 
	    new Epetra_Map( -1, 
			    Teuchos::as<int>(global_rows.size()),
			    global_rows.getRawPtr(),
			    0,
			    *epetra_comm ) );

	return Teuchos::rcp( new Epetra_Vector(*map) );
    }

    /*!
     * \brief Create a deep copy of the provided vector and return a
     * reference-counted pointer.
     */
    static Teuchos::RCP<vector_type> deepCopy( const vector_type& vector )
    {
	return Teuchos::rcp( new Epetra_Vector(vector) );
    }

    /*! 
     * \brief Given a multivector, get a single non-const vector of a given
     * id.
     */
    static Teuchos::RCP<vector_type> getVectorNonConst( 
        multivector_type& multivector, const int id )
    { 
        return Teuchos::rcp( new Epetra_Vector(*multivector(id)) );
    }

    /*!
     * \brief Get the global length of the vector.
     */
    static global_ordinal_type getGlobalLength( const vector_type& vector )
    { 
	return Teuchos::as<global_ordinal_type>( vector.GlobalLength() );
    }

    /*!
     * \brief Get the local length of the vector.
     */
    static local_ordinal_type getLocalLength( const vector_type& vector )
    { 
	return Teuchos::as<local_ordinal_type>( vector.MyLength() );
    }

    /*!
     * \brief Determine whether or not a given global row is on-process.
     */
    static bool isGlobalRow( const vector_type& vector,
			     const global_ordinal_type& global_row )
    {
	return vector.Map().MyGID( global_row );
    }

    /*!
     * \brief Determine whether or not a given local row is on-process.
     */
    static bool isLocalRow( const vector_type& vector,
			    const local_ordinal_type& local_row )
    {
	return vector.Map().MyLID( local_row );
    }

    /*!
     * \brief Given a local row on-process, provide the global ordinal.
     */
    static global_ordinal_type 
    getGlobalRow( const vector_type& vector, 
		  const local_ordinal_type& local_row )
    { 
	return vector.Map().GID( local_row );
    }

    /*!
     * \brief Given a global row on-process, provide the local ordinal.
     */
    static local_ordinal_type 
    getLocalRow( const vector_type& vector,
		 const global_ordinal_type& global_row )
    { 
	return vector.Map().LID( global_row );
    }

    /*!
     * \brief Replace value at the global row index. The global index must
     * exist on process.
     */
    static void replaceGlobalValue( vector_type& vector,
    				    global_ordinal_type global_row,
    				    const scalar_type& value )
    {
	MCLS_CHECK_ERROR_CODE(
	    vector.ReplaceGlobalValue( global_row, 0, value )
	    );
    }

    /*!
     * \brief Replace value at the local row index. The local index must exist
     * on process.
     */
    static void replaceLocalValue( vector_type& vector,
    				   local_ordinal_type local_row,
    				   const scalar_type& value )
    {
	MCLS_CHECK_ERROR_CODE(
	    vector.ReplaceMyValue( local_row, 0, value )
	    );
    }

    /*!
     * \brief Sum a value into existing value at the global row index. The
     * global index must exist on process.
     */
    static void sumIntoGlobalValue( vector_type& vector,
    				    global_ordinal_type global_row,
    				    const scalar_type& value )
    {
	MCLS_CHECK_ERROR_CODE(
	    vector.SumIntoGlobalValue( global_row, 0, value )
	    );
    }

    /*!
     * \brief Sum a value into existing value at the local row index. The
     * local index must exist on process.
     */
    static void sumIntoLocalValue( vector_type& vector,
    				   local_ordinal_type local_row,
    				   const scalar_type& value )
    {
	MCLS_CHECK_ERROR_CODE(
	    vector.SumIntoMyValue( local_row, 0, value )
	    );
    }

    /*!
     * \brief Set all values in the vector to a given value.
     */
    static void putScalar( vector_type& vector, const scalar_type& value )
    { 
	MCLS_CHECK_ERROR_CODE(
	    vector.PutScalar( value )
	    );
    }

    /*!
     * \brief Fill the vector with random values.
     */
    static void randomize( vector_type& vector )
    {
        MCLS_CHECK_ERROR_CODE(
	    vector.Random()
	    );
    }

    /*!
     * \brief Get a const view of the local vector data.
     */
    static Teuchos::ArrayRCP<const scalar_type> view( const vector_type& vector )
    { 
	scalar_type* view_pointer;
	MCLS_CHECK_ERROR_CODE(
	    vector.ExtractView( &view_pointer )
	    );
	return Teuchos::ArrayRCP<const scalar_type>( 
	    view_pointer, 0, vector.MyLength(), false );
    }

    /*!
     * \brief Get a non-const view of the local vector data.
     */
    static Teuchos::ArrayRCP<scalar_type> 
    viewNonConst( vector_type& vector )
    { 
	scalar_type* view_pointer;
	MCLS_CHECK_ERROR_CODE(
	    vector.ExtractView( &view_pointer )
	    );
	return Teuchos::ArrayRCP<scalar_type>( 
	    view_pointer, 0, vector.MyLength(), false );
    }

    /*!
     * \brief Compute the dot product of two vectors A \dot B.
     */
    static scalar_type 
    dot( const vector_type& A, const vector_type& B )
    { 
	scalar_type product;
	MCLS_CHECK_ERROR_CODE(
	    A.Dot( B, &product )
	    );
	return product;
    }

    /*! 
     * \brief Compute the 1-norm of a vector.
     */
    static Teuchos::ScalarTraits<scalar_type>::magnitudeType 
    norm1( const vector_type& vector )
    {
	scalar_type norm;
	MCLS_CHECK_ERROR_CODE(
	    vector.Norm1( &norm )
	    );
	return norm;
    }

    /*! 
     * \brief Compute the 2-norm of a vector.
     */
    static Teuchos::ScalarTraits<scalar_type>::magnitudeType 
    norm2( const vector_type& vector )
    {
	scalar_type norm;
	MCLS_CHECK_ERROR_CODE(
	    vector.Norm2( &norm )
	    );
	return norm;
    }

    /*! 
     * \brief Compute the Inf-norm of a vector.
     */
    static Teuchos::ScalarTraits<scalar_type>::magnitudeType 
    normInf( const vector_type& vector )
    {
	scalar_type norm;
	MCLS_CHECK_ERROR_CODE(
	    vector.NormInf( &norm )
	    );
	return norm;
    }

    /*!
     * \brief Compute the mean value of a vector.
     */
    static scalar_type meanValue( const vector_type& vector )
    {
	scalar_type mean;
	MCLS_CHECK_ERROR_CODE(
	    vector.MeanValue( &mean )
	    );
	return mean;
    }

    /*!
     * \brief Replace output vector values with element-wise absolute values
     * of input vector A = abs(B).
     */
    static void abs( vector_type& A, const vector_type& B )
    {
	MCLS_CHECK_ERROR_CODE(
	    A.Abs( B )
	    );
    }

    /*!
     * \brief Scale a vector by a value A = value*A.
     */
    static void scale( vector_type& A, const scalar_type value )
    {
	MCLS_CHECK_ERROR_CODE(
	    A.Scale( value )
	    );
    }

    /*!
     * \brief Scale a vector by a value A = value*B.
     */
    static void 
    scaleCopy( vector_type& A, const scalar_type value, const vector_type& B )
    {
	MCLS_CHECK_ERROR_CODE(
	    A.Scale( value, B )
	    );
    }
    
    /*!
     * \brief Replace output vector values with element-wise reciprocal values
     * of input vector A(i) = 1 / B(i)
     */
    static void reciprocal( vector_type& A, const vector_type& B )
    { 
	MCLS_CHECK_ERROR_CODE(
	    A.Reciprocal( B )
	    );
    }

    /*!
     * \brief Update vector with A = alpha*A + beta*B.
     */
    static void update( vector_type& A, const scalar_type& alpha,
    			const vector_type& B, const scalar_type& beta )
    {
	MCLS_CHECK_ERROR_CODE(
	    A.Update( beta, B, alpha )
	    );
    }

    /*!
     * \brief Update vector with A = alpha*A + beta*B + gamma*C.
     */
    static void update( vector_type& A, const scalar_type& alpha,
    			const vector_type& B, const scalar_type& beta,
    			const vector_type& C, const scalar_type& gamma )
    {
	MCLS_CHECK_ERROR_CODE(
	    A.Update( beta, B, gamma, C, alpha )
	    );
    }

    /*!
     * \brief Element-wise mulitply two vectors 
     * A(i) = alpha*A(i) + beta*B(i)*C(i).
     */
    static void elementWiseMultiply( vector_type& A, const scalar_type& alpha,
				     const vector_type& B, const vector_type& C, 
				     const scalar_type& beta)
    { 
	MCLS_CHECK_ERROR_CODE(
	    A.Multiply( beta, B, C, alpha )
	    );
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_EPETRAVECTORADAPTER_HPP

//---------------------------------------------------------------------------//
// end MCLS_EpetraVectorAdapater.hpp
//---------------------------------------------------------------------------//
