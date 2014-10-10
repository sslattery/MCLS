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

#include <Thyra_VectorBase.hpp>
#include <Thyra_VectorSpaceBase.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
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
template<class Vector, class Matrix>
class ThyraVectorExtraction
{
  public:

    typedef Vector                            vector_type;
    typedef typename vector_type::scalar_type scalar_type;
    typedef Matrix                            matrix_type;

    static Teuchos::RCP<const Thyra::VectorSpaceBase<scalar_type> >
    createVectorSpaceFromDomain( const matrix_type& matrix )
    {
	UndefinedThyraVectorExtraction<vector_type>::notDefined();
	return Teuchos::null;
    }

    static Teuchos::RCP<const Thyra::VectorSpaceBase<scalar_type> >
    createVectorSpaceFromRange( const matrix_type& matrix )
    {
	UndefinedThyraVectorExtraction<vector_type>::notDefined();
	return Teuchos::null;
    }

    static Teuchos::RCP<vector_type>
    getVectorFromDomain( 
	const Teuchos::RCP<Thyra::VectorBase<scalar_type> >& thyra_vector,
	const matrix_type& matrix )
    {
	UndefinedThyraVectorExtraction<vector_type>::notDefined();
	return Teuchos::null;
    }

    static Teuchos::RCP<vector_type>
    getVectorFromRange( 
	const Teuchos::RCP<Thyra::VectorBase<scalar_type> >& thyra_vector,
	const matrix_type& matrix )
    {
	UndefinedThyraVectorExtraction<vector_type>::notDefined();
	return Teuchos::null;
    }

    static Teuchos::RCP<const vector_type>
    getVectorNonConstFromDomain( 
	const Teuchos::RCP<const Thyra::VectorBase<scalar_type> >& thyra_vector,
	const matrix_type& matrix )
    {
	UndefinedThyraVectorExtraction<vector_type>::notDefined();
	return Teuchos::null;
    }

    static Teuchos::RCP<const vector_type>
    getVectorNonConstFromRange( 
	const Teuchos::RCP<const Thyra::VectorBase<scalar_type> >& thyra_vector,
	const matrix_type& matrix )
    {
	UndefinedThyraVectorExtraction<vector_type>::notDefined();
	return Teuchos::null;
    }

    static Teuchos::RCP<Thyra::VectorBase<scalar_type> >
    createThyraVectorFromDomain( const Teuchos::RCP<vector_type>& vector,
				 const matrix_type& matrix )
    {
	UndefinedThyraVectorExtraction<vector_type>::notDefined();
	return Teuchos::null;
    }

    static Teuchos::RCP<Thyra::VectorBase<scalar_type> >
    createThyraVectorFromRange( const Teuchos::RCP<vector_type>& vector,
				const matrix_type& matrix )
    {
	UndefinedThyraVectorExtraction<vector_type>::notDefined();
	return Teuchos::null;
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_THYRAVECTOREXTRACTION_HPP

//---------------------------------------------------------------------------//
// end MCLS_ThyraVectorExtraction.hpp
//---------------------------------------------------------------------------//

