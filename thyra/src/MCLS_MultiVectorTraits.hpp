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
 * \file MCLS_MultiVectorTraits.hpp
 * \author Stuart R. Slattery
 * \brief MultiVector traits definition.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MULTIVECTORTRAITS_HPP
#define MCLS_MULTIVECTORTRAITS_HPP

#include <MCLS_VectorTraits.hpp>
#include <MCLS_TpetraAdapter.hpp>
#include <MCLS_EpetraAdapter.hpp>

#include <Teuchos_RCP.hpp>

#include <Epetra_Comm.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_MultiVector.h>

#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>

#include <Thyra_MultiVectorBase.hpp>
#include <Thyra_SpmdVectorSpaceBase.hpp>
#include <Thyra_EpetraThyraWrappers.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class UndefinedMultiVectorTraits
 * \brief Class for undefined vector traits. 
 *
 * Will throw a compile-time error if these traits are not specialized.
 */
template<class MultiVector>
struct UndefinedMultiVectorTraits
{
    static inline void notDefined()
    {
	return MultiVector::this_type_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class MultiVectorTraits
 * \brief Traits for vectors.
 *
 * MultiVectorTraits defines an interface for parallel distributed vectors
 * (e.g. Tpetra::MultiVector or Epetra_MultiVector).
 */
template<class MultiVector>
class MultiVectorTraits
{
  public:

    //@{
    //! Typedefs.
    typedef MultiVector                                      multivector_type;
    typedef typename MultiVector::vector_type                vector_type;
    typedef typename MultiVector::scalar_type                scalar_type;
    typedef typename MultiVector::local_ordinal_type         local_ordinal_type;
    typedef typename MultiVector::global_ordinal_type        global_ordinal_type;
    //@}

    //! Get the number of vectors in this multivector.
    static int getNumVectors( const MultiVector& multivector )
    { 
	UndefinedMultiVectorTraits<MultiVector>::notDefined(); 
	return 0;
    }

    //! Return a vector given its id in the multivector.
    static Teuchos::RCP<const vector_type> getVector( const int id )
    {
	UndefinedMultiVectorTraits<MultiVector>::notDefined(); 
	return Teuchos::null;
    }

    //! Return a vector given its id in the multivector.
    static Teuchos::RCP<vector_type> getVectorNonConst( const int id )
    {
	UndefinedMultiVectorTraits<MultiVector>::notDefined(); 
	return Teuchos::null;
    }

    //! Return a multivector given a Thyra base multivector.
    static Teuchos::RCP<multivector_type>
    getMultiVectorFromThyra( 
	const Teuchos::RCP<Thyra::MultiVectorBase<scalar_type> >& thyra_mv )
    {
	UndefinedMultiVectorTraits<MultiVector>::notDefined(); 
	return Teuchos::null;
    }

    //! Return a multivector given a Thyra base multivector.
    static Teuchos::RCP<const multivector_type>
    getConstMultiVectorFromThyra( 
	const Teuchos::RCP<const Thyra::MultiVectorBase<scalar_type> >& thyra_mv )
    {
	UndefinedMultiVectorTraits<MultiVector>::notDefined(); 
	return Teuchos::null;
    }
};

//---------------------------------------------------------------------------//
// Specialization for Epetra_MultiVector.
//---------------------------------------------------------------------------//
template<>
class MultiVectorTraits<Epetra_MultiVector>
{
  public:

    //@{
    //! Typedefs.
    typedef Epetra_MultiVector                          multivector_type;
    typedef Epetra_Vector                               vector_type;
    typedef VectorTraits<vector_type>                   VT;
    typedef VT::scalar_type                             scalar_type;
    typedef VT::local_ordinal_type                      local_ordinal_type;
    typedef VT::global_ordinal_type                     global_ordinal_type;
    //@}

    //! Get the number of vectors in this multivector.
    static int getNumVectors( const multivector_type& multivector )
    { 
	return multivector.NumVectors();
    }

    //! Return a vector given its id in the multivector.
    static Teuchos::RCP<const vector_type> 
    getVector( const multivector_type& multivector, const int id )
    {
	return Teuchos::rcp( multivector(id), false );
    }

    //! Return a vector given its id in the multivector.
    static Teuchos::RCP<vector_type> 
    getVectorNonConst( multivector_type& multivector, const int id )
    {
	return Teuchos::rcp( multivector(id), false );
    }

    //! Return a multivector given a Thyra base multivector.
    static Teuchos::RCP<multivector_type>
    getMultiVectorFromThyra( 
	const Teuchos::RCP<Thyra::MultiVectorBase<scalar_type> >& thyra_mv )
    {
	Teuchos::RCP<const Thyra::SpmdVectorSpaceBase<scalar_type> > space =
	    Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorSpaceBase<scalar_type> >(
		thyra_mv->col(0)->space() );
	Teuchos::RCP<const Epetra_Comm> comm = 
	    Thyra::get_Epetra_Comm( *space->getComm() );
	Teuchos::RCP<const Epetra_Map> map =
	    Thyra::get_Epetra_Map( *space, comm );
	return Thyra::get_Epetra_MultiVector( *map, thyra_mv );
    }

    //! Return a multivector given a Thyra base multivector.
    static Teuchos::RCP<const multivector_type>
    getConstMultiVectorFromThyra( 
	const Teuchos::RCP<const Thyra::MultiVectorBase<scalar_type> >& thyra_mv )
    {
	Teuchos::RCP<const Thyra::SpmdVectorSpaceBase<scalar_type> > space =
	    Teuchos::rcp_dynamic_cast<const Thyra::SpmdVectorSpaceBase<scalar_type> >(
		thyra_mv->col(0)->space() );
	Teuchos::RCP<const Epetra_Comm> comm = 
	    Thyra::get_Epetra_Comm( *space->getComm() );
	Teuchos::RCP<const Epetra_Map> map =
	    Thyra::get_Epetra_Map( *space, comm );
	return Thyra::get_Epetra_MultiVector( *map, thyra_mv );
    }
};

//---------------------------------------------------------------------------//
// Specialization for Tpetra::MultiVector.
//---------------------------------------------------------------------------//
template<class Scalar, class LO, class GO>
class MultiVectorTraits<Tpetra::MultiVector<Scalar,LO,GO> >
{
  public:

    //@{
    //! Typedefs.
    typedef Tpetra::MultiVector<Scalar,LO,GO>           multivector_type;
    typedef Tpetra::Vector<Scalar,LO,GO>                vector_type;
    typedef VectorTraits<vector_type>                   VT;
    typedef typename VT::scalar_type                    scalar_type;
    typedef typename VT::local_ordinal_type             local_ordinal_type;
    typedef typename VT::global_ordinal_type            global_ordinal_type;
    //@}

    //! Get the number of vectors in this multivector.
    static int getNumVectors( const multivector_type& multivector )
    { 
	return multivector.getNumVectors();
    }

    //! Return a vector given its id in the multivector.
    static Teuchos::RCP<const vector_type> 
    getVector( const multivector_type& multivector, const int id )
    {
	return multivector.getVector(id);
    }

    //! Return a vector given its id in the multivector.
    static Teuchos::RCP<vector_type> 
    getVectorNonConst( multivector_type& multivector, const int id )
    {
	return multivector.getVectorNonConst(id);
    }

    //! Return a multivector given a Thyra base multivector.
    static Teuchos::RCP<multivector_type>
    getMultiVectorFromThyra( 
	const Teuchos::RCP<Thyra::MultiVectorBase<scalar_type> >& thyra_mv )
    {
	return Thyra::TpetraOperatorVectorExtraction<
	    Scalar,LO,GO>::getTpetraMultiVector( thyra_mv );
    }

    //! Return a multivector given a Thyra base multivector.
    static Teuchos::RCP<const multivector_type>
    getConstMultiVectorFromThyra( 
	const Teuchos::RCP<const Thyra::MultiVectorBase<scalar_type> >& thyra_mv )
    {
	return Thyra::TpetraOperatorVectorExtraction<
	    Scalar,LO,GO>::getConstTpetraMultiVector( thyra_mv );
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_MULTIVECTORTRAITS_HPP

//---------------------------------------------------------------------------//
// end MCLS_MultiVectorTraits.hpp
//---------------------------------------------------------------------------//

