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
 * \file MCLS_ThyraVectorExtraction.hpp
 * \author Stuart R. Slattery
 * \brief Thyra vector extraction utilities.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_THYRAVECTOREXTRACTION_HPP
#define MCLS_THYRAVECTOREXTRACTION_HPP

#include <Teuchos_RCP.hpp>

#include <Epetra_Vector.h>

#include <Tpetra_Vector.hpp>

#include <Thyra_VectorBase.hpp>
#include <Thyra_VectorSpaceBase.hpp>
#include <Thyra_EpetraThyraWrappers.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>

namespace MCLS
{

/*!
 * \class UndefinedThyraVectorExtraction
 * \brief Class for undefined vector extraction.
 *
 * Will throw a compile-time error if these traits are not specialized.
 */
template<class Vector>
struct UndefinedThyraVectorExtraction
{
    static inline void notDefined()
    {
	return Vector::this_type_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class ThyraVectorExtraction
 */
template<class Vector>
class ThyraVectorExtraction
{
  public:

    typedef Vector                            vector_type;
    typedef typename vector_type::scalar_type scalar_type;

    static Teuchos::RCP<const Thyra::VectorSpaceBase<scalar_type> >
    createVectorSpace( const vector_type& vector )
    {
	UndefinedThyraVectorExtraction<vector_type>::notDefined();
	return Teuchos::null;
    }

    static Teuchos::RCP<vector_type>
    getVector( const Teuchos::RCP<Thyra::VectorBase<scalar_type> >& thyra_vector,
	       const vector_type& vector )
    {
	UndefinedThyraVectorExtraction<vector_type>::notDefined();
	return Teuchos::null;
    }

    static Teuchos::RCP<const vector_type>
    getVectorNonConst( const Teuchos::RCP<const Thyra::VectorBase<scalar_type> >& thyra_vector,
	       const vector_type& vector )
    {
	UndefinedThyraVectorExtraction<vector_type>::notDefined();
	return Teuchos::null;
    }

    static Teuchos::RCP<Thyra::VectorBase<scalar_type> >
    createThyraVector( const vector_type& vector )
    {
	UndefinedThyraVectorExtraction<vector_type>::notDefined();
	return Teuchos::null;
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class Epetra specialization.
 */
template<>
class ThyraVectorExtraction<Epetra_Vector>
{
  public:

    typedef Epetra_Vector vector_type;
    typedef double        scalar_type;

    static Teuchos::RCP<const Thyra::VectorSpaceBase<scalar_type> >
    createVectorSpace( const vector_type& vector )
    {
	return Thyra::create_VectorSpace( 
	    Teuchos::rcp_dynamic_cast<const Epetra_Map>(Teuchos::rcpFromRef(vector.Map())) );
    }

    static Teuchos::RCP<vector_type>
    getVectorNonConst( const Teuchos::RCP<Thyra::VectorBase<scalar_type> >& thyra_vector,
	       const vector_type& vector )
    {
	return Thyra::get_Epetra_Vector( 
	    *Teuchos::rcp_dynamic_cast<const Epetra_Map>(Teuchos::rcpFromRef(vector.Map())),
	    thyra_vector );
    }

    static Teuchos::RCP<const vector_type>
    getVector( const Teuchos::RCP<const Thyra::VectorBase<scalar_type> >& thyra_vector,
	       const vector_type& vector )
    {
	return Thyra::get_Epetra_Vector( 
	    *Teuchos::rcp_dynamic_cast<const Epetra_Map>(Teuchos::rcpFromRef(vector.Map())),
	    thyra_vector );
    }

    static Teuchos::RCP<Thyra::VectorBase<scalar_type> >
    createThyraVector( const Teuchos::RCP<vector_type>& vector )
    {
	return Thyra::create_Vector( vector, createVectorSpace(*vector) );
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class Tpetra specialization
 */
template<class Scalar, class LO, class GO>
class ThyraVectorExtraction<Tpetra::Vector<Scalar,LO,GO> >
{
  public:

    typedef Tpetra::Vector<Scalar,LO,GO>      vector_type;
    typedef typename vector_type::scalar_type scalar_type;

    static Teuchos::RCP<const Thyra::VectorSpaceBase<scalar_type> >
    createVectorSpace( const vector_type& vector )
    {
	return Thyra::createVectorSpace<Scalar>( vector.getMap() );
    }

    static Teuchos::RCP<vector_type>
    getVectorNonConst( const Teuchos::RCP<Thyra::VectorBase<scalar_type> >& thyra_vector,
	       const vector_type& vector )
    {
	return Thyra::TpetraOperatorVectorExtraction<Scalar,LO,GO>::getTpetraVector(
	    thyra_vector );
    }

    static Teuchos::RCP<const vector_type>
    getVector( const Teuchos::RCP<const Thyra::VectorBase<scalar_type> >& thyra_vector,
	       const vector_type& vector )
    {
	return Thyra::TpetraOperatorVectorExtraction<Scalar,LO,GO>::getConstTpetraVector(
	    thyra_vector );
    }

    static Teuchos::RCP<Thyra::VectorBase<scalar_type> >
    createThyraVector( const Teuchos::RCP<vector_type>& vector )
    {
	return Thyra::createVector( vector, createVectorSpace(*vector) );
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_THYRAVECTOREXTRACTION_HPP

//---------------------------------------------------------------------------//
// end MCLS_ThyraVectorExtraction.hpp
//---------------------------------------------------------------------------//

