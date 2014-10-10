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
 * \file tstTpetraVector.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra vector tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <string>
#include <cassert>

#include <MCLS_VectorTraits.hpp>
#include <MCLS_TpetraAdapter.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

//---------------------------------------------------------------------------//
// Instantiation macro. 
// 
// These types are those enabled by Tpetra under explicit instantiation.
//---------------------------------------------------------------------------//
#define UNIT_TEST_INSTANTIATION( type, name )			           \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( type, name, int, int, double )   \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( type, name, int, long, double )

//---------------------------------------------------------------------------//
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorTraits, Typedefs, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef typename VT::scalar_type scalar_type;
    typedef typename VT::local_ordinal_type local_ordinal_type;
    typedef typename VT::global_ordinal_type global_ordinal_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<Scalar, scalar_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<LO, local_ordinal_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<GO, global_ordinal_type>::value)
	== true, true );
}

UNIT_TEST_INSTANTIATION( VectorTraits, Typedefs )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorTraits, Clone, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map );

    VT::putScalar( *A, 1.0 );

    Teuchos::RCP<VectorType> B = VT::clone( *A );

    TEST_ASSERT( A->getMap()->isSameAs( *(B->getMap()) ) );
    
    Teuchos::ArrayRCP<const Scalar> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator view_iterator;
    for ( view_iterator = B_view.begin();
	  view_iterator != B_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 0.0 );
    }
}

UNIT_TEST_INSTANTIATION( VectorTraits, Clone )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorTraits, RowCreate, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map );

    Teuchos::ArrayView<const GO> rows = map->getNodeElementList();
    Teuchos::RCP<VectorType> B = VT::createFromRows( comm, rows );

    TEST_ASSERT( A->getMap()->isSameAs( *(B->getMap()) ) );
    
    Teuchos::ArrayRCP<const Scalar> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator view_iterator;
    for ( view_iterator = B_view.begin();
	  view_iterator != B_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 0.0 );
    }
}

UNIT_TEST_INSTANTIATION( VectorTraits, RowCreate )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorTraits, DeepCopy, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map );
    VT::putScalar( *A, 1.0 );

    Teuchos::RCP<VectorType> B = VT::deepCopy( *A );

    TEST_ASSERT( A->getMap()->isSameAs( *(B->getMap()) ) );
    
    Teuchos::ArrayRCP<const Scalar> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator view_iterator;
    for ( view_iterator = B_view.begin();
	  view_iterator != B_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 1.0 );
    }
}

UNIT_TEST_INSTANTIATION( VectorTraits, DeepCopy )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorTraits, Modifiers, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map );

    VT::putScalar( *A, 2.0 );    

    Teuchos::ArrayRCP<const Scalar> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator view_iterator;
    for ( view_iterator = A_view.begin();
	  view_iterator != A_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 2.0 );
    }

    Teuchos::ArrayRCP<Scalar> A_view_non_const = VT::viewNonConst( *A );
    typename Teuchos::ArrayRCP<Scalar>::iterator view_non_const_iterator;
    for ( view_non_const_iterator = A_view_non_const.begin();
	  view_non_const_iterator != A_view_non_const.end();
	  ++view_non_const_iterator )
    {
	*view_non_const_iterator = 3.0;
    }

    for ( view_iterator = A_view.begin();
	  view_iterator != A_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 3.0 );
    }
}

UNIT_TEST_INSTANTIATION( VectorTraits, Modifiers )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorTraits, SumIntoElement, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map );
    VT::putScalar( *A, 1.0 );

    Teuchos::ArrayView<const GO> global_elements = 
	A->getMap()->getNodeElementList();
    typename Teuchos::ArrayView<const GO>::const_iterator element_iterator;
    for ( element_iterator = global_elements.begin();
	  element_iterator != global_elements.end();
	  ++element_iterator )
    {
	int local_element = VT::getLocalRow( *A, *element_iterator );
	int global_row = VT::getGlobalRow( *A, local_element );
	TEST_ASSERT( VT::isGlobalRow( *A, global_row ) );
	VT::sumIntoGlobalValue( *A, global_row, 2.0 );
    }

    Teuchos::ArrayRCP<const Scalar> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator view_iterator;
    for ( view_iterator = A_view.begin();
	  view_iterator != A_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 3.0 );
    }

    for ( element_iterator = global_elements.begin();
	  element_iterator != global_elements.end();
	  ++element_iterator )
    {
	LO local_element = VT::getLocalRow( *A, *element_iterator );
	TEST_ASSERT( VT::isLocalRow( *A, local_element ) );
	VT::sumIntoLocalValue( *A, local_element, 2.0 );
    }

    for ( view_iterator = A_view.begin();
	  view_iterator != A_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 5.0 );
    }
}

UNIT_TEST_INSTANTIATION( VectorTraits, SumIntoElement )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorTraits, ReplaceElement, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map );
    VT::putScalar( *A, 1.0 );

    Teuchos::ArrayView<const GO> global_elements = 
	A->getMap()->getNodeElementList();
    typename Teuchos::ArrayView<const GO>::const_iterator element_iterator;
    for ( element_iterator = global_elements.begin();
	  element_iterator != global_elements.end();
	  ++element_iterator )
    {
	int local_element = VT::getLocalRow( *A, *element_iterator );
	int global_row = VT::getGlobalRow( *A, local_element );
	TEST_ASSERT( VT::isGlobalRow( *A, global_row ) );
	VT::replaceGlobalValue( *A, global_row, 2.0 );
    }

    Teuchos::ArrayRCP<const Scalar> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator view_iterator;
    for ( view_iterator = A_view.begin();
	  view_iterator != A_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 2.0 );
    }

    for ( element_iterator = global_elements.begin();
	  element_iterator != global_elements.end();
	  ++element_iterator )
    {
	LO local_element = VT::getLocalRow( *A, *element_iterator );
	TEST_ASSERT( VT::isLocalRow( *A, local_element ) );
	VT::replaceLocalValue( *A, local_element, 5.0 );
    }

    for ( view_iterator = A_view.begin();
	  view_iterator != A_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 5.0 );
    }
}

UNIT_TEST_INSTANTIATION( VectorTraits, ReplaceElement )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorTraits, DotProduct, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map );
    VT::putScalar( *A, 2.0 );

    Teuchos::RCP<VectorType> B = VT::deepCopy( *A );

    Scalar product = 2.0*2.0*global_num_rows;
    TEST_EQUALITY( VT::dot( *A, *B ), product );
}

UNIT_TEST_INSTANTIATION( VectorTraits, DotProduct )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorTraits, Norms, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map );
    VT::putScalar( *A, 2.0 );

    Scalar norm_two = std::pow( 4.0*global_num_rows, 0.5 );
    TEST_EQUALITY( VT::norm2( *A ), norm_two );
    TEST_EQUALITY( VT::norm1( *A ), 2.0*global_num_rows );
    TEST_EQUALITY( VT::normInf( *A ), 2.0 );
}

UNIT_TEST_INSTANTIATION( VectorTraits, Norms )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorTraits, MeanValue, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map );
    VT::putScalar( *A, 2.0 );
    VT::replaceLocalValue( *A, 0, 1.0 );

    Scalar mean_value = ((global_num_rows-comm_size)*2.0 + comm_size*1.0)
			/ global_num_rows;
    TEST_FLOATING_EQUALITY( VT::meanValue( *A ), mean_value, 1.0e-12 );
}

UNIT_TEST_INSTANTIATION( VectorTraits, MeanValue )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorTraits, AbsoluteVal, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map );
    VT::putScalar( *A, -2.0 );

    Teuchos::RCP<VectorType> B = VT::clone( *A );

    VT::abs( *B, *A );
    Teuchos::ArrayRCP<const Scalar> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator B_view_iterator;
    for ( B_view_iterator = B_view.begin();
	  B_view_iterator != B_view.end();
	  ++B_view_iterator )
    {
	TEST_EQUALITY( *B_view_iterator, 2.0 );
    }

    VT::abs( *A, *A );
    Teuchos::ArrayRCP<const Scalar> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator A_view_iterator;
    for ( A_view_iterator = A_view.begin();
	  A_view_iterator != A_view.end();
	  ++A_view_iterator )
    {
	TEST_EQUALITY( *A_view_iterator, 2.0 );
    }
}

UNIT_TEST_INSTANTIATION( VectorTraits, AbsoluteVal )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorTraits, Scale, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map );
    VT::putScalar( *A, 2.0 );

    VT::scale( *A, 3.0 );
    Teuchos::ArrayRCP<const Scalar> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator A_view_iterator;
    for ( A_view_iterator = A_view.begin();
	  A_view_iterator != A_view.end();
	  ++A_view_iterator )
    {
	TEST_EQUALITY( *A_view_iterator, 6.0 );
    }

    Teuchos::RCP<VectorType> B = VT::clone( *A );
    VT::scaleCopy( *B, 2.0, *A );
    Teuchos::ArrayRCP<const Scalar> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator B_view_iterator;
    for ( B_view_iterator = B_view.begin();
	  B_view_iterator != B_view.end();
	  ++B_view_iterator )
    {
	TEST_EQUALITY( *B_view_iterator, 12.0 );
    }
}

UNIT_TEST_INSTANTIATION( VectorTraits, Scale )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorTraits, Reciprocal, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map );
    VT::putScalar( *A, 2.0 );

    Teuchos::RCP<VectorType> B = VT::clone( *A );

    Scalar recip_val = 1 / 2.0;
    VT::reciprocal( *B, *A );
    Teuchos::ArrayRCP<const Scalar> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator B_view_iterator;
    for ( B_view_iterator = B_view.begin();
	  B_view_iterator != B_view.end();
	  ++B_view_iterator )
    {
	TEST_EQUALITY( *B_view_iterator, recip_val );
    }

    VT::reciprocal( *A, *A );
    Teuchos::ArrayRCP<const Scalar> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator A_view_iterator;
    for ( A_view_iterator = A_view.begin();
	  A_view_iterator != A_view.end();
	  ++A_view_iterator )
    {
	TEST_EQUALITY( *A_view_iterator, recip_val );
    }
}

UNIT_TEST_INSTANTIATION( VectorTraits, Reciprocal )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorTraits, Update, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map );
    VT::putScalar( *A, 1.0 );

    Teuchos::RCP<VectorType> B = VT::clone( *A );
    VT::putScalar( *B, 2.0 );

    Scalar alpha = 4.0;
    Scalar beta = 5.0;

    Scalar update_val = alpha*1.0 + beta*2.0;
    VT::update( *A, alpha, *B, beta );
    Teuchos::ArrayRCP<const Scalar> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator A_view_iterator;
    for ( A_view_iterator = A_view.begin();
	  A_view_iterator != A_view.end();
	  ++A_view_iterator )
    {
	TEST_EQUALITY( *A_view_iterator, update_val );
    }

    Teuchos::RCP<VectorType> C = VT::clone( *A );
    VT::putScalar( *C, 3.0 );
    Scalar gamma = 6.0;
    update_val = update_val*alpha + beta*2.0 + gamma*3.0;
    VT::update( *A, alpha, *B, beta, *C, gamma );
    for ( A_view_iterator = A_view.begin();
	  A_view_iterator != A_view.end();
	  ++A_view_iterator )
    {
	TEST_EQUALITY( *A_view_iterator, update_val );
    }
}

UNIT_TEST_INSTANTIATION( VectorTraits, Update )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorTraits, ElementWiseMultiply, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map );
    VT::putScalar( *A, 1.0 );

    Teuchos::RCP<VectorType> B = VT::clone( *A );
    VT::putScalar( *B, 2.0 );

    Teuchos::RCP<VectorType> C = VT::clone( *A );
    VT::putScalar( *C, 3.0 );

    Scalar alpha = 4.0;
    Scalar beta = 5.0;

    Scalar multiply_val = 1.0*alpha + beta*2.0*3.0;
    VT::elementWiseMultiply( *A, alpha, *B, *C, beta );

    Teuchos::ArrayRCP<const Scalar> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator A_view_iterator;
    for ( A_view_iterator = A_view.begin();
	  A_view_iterator != A_view.end();
	  ++A_view_iterator )
    {
	TEST_EQUALITY( *A_view_iterator, multiply_val );
    }
}

UNIT_TEST_INSTANTIATION( VectorTraits, ElementWiseMultiply )

//---------------------------------------------------------------------------//
// end tstTpetraVector.cpp
//---------------------------------------------------------------------------//

