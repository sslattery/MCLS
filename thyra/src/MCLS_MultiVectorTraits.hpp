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
#include <MCLS_DBC.hpp>

#include <Teuchos_RCP.hpp>

#include <Epetra_Comm.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_MultiVector.h>
#include <Epetra_RowMatrix.h>

#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>

#include <Thyra_MultiVectorBase.hpp>
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
template<class MultiVector, class Matrix>
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
template<class MultiVector, class Matrix>
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
    typedef Matrix                                           matrix_type;
    //@}

    //! Given a multivector, construct a deep copy.
    static Teuchos::RCP<multivector_type> 
    deepCopy( const MultiVector& multivector )
    { 
	UndefinedMultiVectorTraits<MultiVector,Matrix>::notDefined(); 
	return Teuchos::null;
    }

    //! Get the number of vectors in this multivector.
    static int getNumVectors( const MultiVector& multivector )
    { 
	UndefinedMultiVectorTraits<MultiVector,Matrix>::notDefined(); 
	return 0;
    }

    //! Return a vector given its id in the multivector.
    static Teuchos::RCP<const vector_type> getVector( const int id )
    {
	UndefinedMultiVectorTraits<MultiVector,Matrix>::notDefined(); 
	return Teuchos::null;
    }

    //! Return a vector given its id in the multivector.
    static Teuchos::RCP<vector_type> getVectorNonConst( const int id )
    {
	UndefinedMultiVectorTraits<MultiVector,Matrix>::notDefined(); 
	return Teuchos::null;
    }

    //! Return a multivector given a Thyra base multivector in the range space
    //! decomposition of the given linear operator.
    static Teuchos::RCP<multivector_type>
    getRangeMultiVectorFromThyra( 
	const Teuchos::RCP<Thyra::MultiVectorBase<scalar_type> >& thyra_mv,
        const Teuchos::RCP<const matrix_type>& matrix )
    {
	UndefinedMultiVectorTraits<MultiVector,Matrix>::notDefined(); 
	return Teuchos::null;
    }

    //! Return a multivector given a Thyra base multivector in the range space
    //! decomposition of the given linear operator.
    static Teuchos::RCP<const multivector_type>
    getConstRangeMultiVectorFromThyra( 
	const Teuchos::RCP<const Thyra::MultiVectorBase<scalar_type> >& thyra_mv,
        const Teuchos::RCP<const matrix_type>& matrix )
    {
	UndefinedMultiVectorTraits<MultiVector,Matrix>::notDefined(); 
	return Teuchos::null;
    }

    //! Return a multivector given a Thyra base multivector in the domain space
    //! decomposition of the given linear operator.
    static Teuchos::RCP<multivector_type>
    getDomainMultiVectorFromThyra( 
	const Teuchos::RCP<Thyra::MultiVectorBase<scalar_type> >& thyra_mv,
        const Teuchos::RCP<const matrix_type>& matrix )
    {
	UndefinedMultiVectorTraits<MultiVector,Matrix>::notDefined(); 
	return Teuchos::null;
    }

    //! Return a multivector given a Thyra base multivector in the domain space
    //! decomposition of the given linear operator.
    static Teuchos::RCP<const multivector_type>
    getConstDomainMultiVectorFromThyra( 
	const Teuchos::RCP<const Thyra::MultiVectorBase<scalar_type> >& thyra_mv,
        const Teuchos::RCP<const matrix_type>& matrix )
    {
	UndefinedMultiVectorTraits<MultiVector,Matrix>::notDefined(); 
	return Teuchos::null;
    }
};

//---------------------------------------------------------------------------//
// Specialization for Epetra_MultiVector.
//---------------------------------------------------------------------------//
template<>
class MultiVectorTraits<Epetra_MultiVector, Epetra_RowMatrix>
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
    typedef Epetra_RowMatrix                             matrix_type;
    //@}

    //! Given a multivector, construct a deep copy.
    static Teuchos::RCP<multivector_type> 
    deepCopy( const multivector_type& multivector )
    { 
	return Teuchos::rcp( new multivector_type(multivector) );
    }

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

    //! Return a multivector given a Thyra base multivector in the range space
    //! decomposition of the given linear operator.
    static Teuchos::RCP<multivector_type>
    getRangeMultiVectorFromThyra( 
	const Teuchos::RCP<Thyra::MultiVectorBase<scalar_type> >& thyra_mv,
        const Teuchos::RCP<const matrix_type>& matrix )
    {
	MCLS_REQUIRE( Teuchos::nonnull(thyra_mv) );
        const Epetra_Map& range_map = matrix->OperatorRangeMap();
        Teuchos::RCP<multivector_type> epetra_mv =
            Thyra::get_Epetra_MultiVector( range_map, thyra_mv );
        MCLS_ENSURE( Teuchos::nonnull(epetra_mv) );
        return epetra_mv;
    }

    //! Return a multivector given a Thyra base multivector in the range space
    //! decomposition of the given linear operator.
    static Teuchos::RCP<const multivector_type>
    getConstRangeMultiVectorFromThyra( 
	const Teuchos::RCP<const Thyra::MultiVectorBase<scalar_type> >& thyra_mv,
        const Teuchos::RCP<const matrix_type>& matrix )
    {
	MCLS_REQUIRE( Teuchos::nonnull(thyra_mv) );
        const Epetra_Map& range_map = matrix->OperatorRangeMap();
        Teuchos::RCP<const multivector_type> epetra_mv = 
            Thyra::get_Epetra_MultiVector( range_map, thyra_mv );
        MCLS_ENSURE( Teuchos::nonnull(epetra_mv) );
        return epetra_mv;
    }

    //! Return a multivector given a Thyra base multivector in the domain space
    //! decomposition of the given linear operator.
    static Teuchos::RCP<multivector_type>
    getDomainMultiVectorFromThyra( 
	const Teuchos::RCP<Thyra::MultiVectorBase<scalar_type> >& thyra_mv,
        const Teuchos::RCP<const matrix_type>& matrix )
    {
	MCLS_REQUIRE( Teuchos::nonnull(thyra_mv) );
        const Epetra_Map& domain_map = matrix->OperatorDomainMap();
        Teuchos::RCP<multivector_type> epetra_mv = 
            Thyra::get_Epetra_MultiVector( domain_map, thyra_mv );
        MCLS_ENSURE( Teuchos::nonnull(epetra_mv) );
        return epetra_mv;
    }

    //! Return a multivector given a Thyra base multivector in the domain space
    //! decomposition of the given linear operator.
    static Teuchos::RCP<const multivector_type>
    getConstDomainMultiVectorFromThyra( 
	const Teuchos::RCP<const Thyra::MultiVectorBase<scalar_type> >& thyra_mv,
        const Teuchos::RCP<const matrix_type>& matrix )
    {
	MCLS_REQUIRE( Teuchos::nonnull(thyra_mv) );
        const Epetra_Map& domain_map = matrix->OperatorDomainMap();
        Teuchos::RCP<const multivector_type> epetra_mv = 
            Thyra::get_Epetra_MultiVector( domain_map, thyra_mv );
        MCLS_ENSURE( Teuchos::nonnull(epetra_mv) );
        return epetra_mv;
    }
};

//---------------------------------------------------------------------------//
// Specialization for Tpetra::MultiVector.
//---------------------------------------------------------------------------//
template<class Scalar, class LO, class GO>
class MultiVectorTraits<Tpetra::MultiVector<Scalar,LO,GO>,
                        Tpetra::CrsMatrix<Scalar,LO,GO> >
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
    typedef Tpetra::CrsMatrix<Scalar,LO,GO>             matrix_type;
    //@}

    //! Given a multivector, construct a deep copy.
    static Teuchos::RCP<multivector_type> 
    deepCopy( const multivector_type& multivector )
    { 
	return Teuchos::rcp( new multivector_type(multivector) );
    }

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
    getRangeMultiVectorFromThyra(
	const Teuchos::RCP<Thyra::MultiVectorBase<scalar_type> >& thyra_mv,
        const Teuchos::RCP<const matrix_type>& matrix = Teuchos::null )
    {
	MCLS_REQUIRE( Teuchos::nonnull(thyra_mv) );
	return Thyra::TpetraOperatorVectorExtraction<
	    Scalar,LO,GO>::getTpetraMultiVector( thyra_mv );
    }

    //! Return a multivector given a Thyra base multivector.
    static Teuchos::RCP<const multivector_type>
    getConstRangeMultiVectorFromThyra( 
	const Teuchos::RCP<const Thyra::MultiVectorBase<scalar_type> >& thyra_mv,
        const Teuchos::RCP<const matrix_type>& matrix = Teuchos::null )
    {
	MCLS_REQUIRE( Teuchos::nonnull(thyra_mv) );
	return Thyra::TpetraOperatorVectorExtraction<
	    Scalar,LO,GO>::getConstTpetraMultiVector( thyra_mv );
    }

    //! Return a multivector given a Thyra base multivector.
    static Teuchos::RCP<multivector_type>
    getDomainMultiVectorFromThyra( 
	const Teuchos::RCP<Thyra::MultiVectorBase<scalar_type> >& thyra_mv,
        const Teuchos::RCP<const matrix_type>& matrix = Teuchos::null )
    {
	MCLS_REQUIRE( Teuchos::nonnull(thyra_mv) );
	return Thyra::TpetraOperatorVectorExtraction<
	    Scalar,LO,GO>::getTpetraMultiVector( thyra_mv );
    }

    //! Return a multivector given a Thyra base multivector.
    static Teuchos::RCP<const multivector_type>
    getConstDomainMultiVectorFromThyra( 
	const Teuchos::RCP<const Thyra::MultiVectorBase<scalar_type> >& thyra_mv,
        const Teuchos::RCP<const matrix_type>& matrix = Teuchos::null )
    {
	MCLS_REQUIRE( Teuchos::nonnull(thyra_mv) );
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

