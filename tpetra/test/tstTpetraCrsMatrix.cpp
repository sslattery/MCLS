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
 * \file tstTpetraCrsMatrix.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra::CrsMatrix adapter tests.
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

#include <MCLS_MatrixTraits.hpp>
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
#include <Tpetra_CrsMatrix.hpp>

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
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, Typedefs, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

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

UNIT_TEST_INSTANTIATION( MatrixTraits, Typedefs )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, RowVectorClone, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );

    Teuchos::RCP<VectorType> X = MT::cloneVectorFromMatrixRows( *A );

    TEST_ASSERT( A->getRowMap()->isSameAs( *(X->getMap()) ) );

    Teuchos::ArrayRCP<const Scalar> X_view = VT::view( *X );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator view_iterator;
    for ( view_iterator = X_view.begin();
	  view_iterator != X_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 0.0 );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, RowVectorClone )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, ColVectorClone, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    Teuchos::RCP<VectorType> X = MT::cloneVectorFromMatrixCols( *A );

    TEST_ASSERT( A->getColMap()->isSameAs( *(X->getMap()) ) );

    Teuchos::ArrayRCP<const Scalar> X_view = VT::view( *X );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator view_iterator;
    for ( view_iterator = X_view.begin();
	  view_iterator != X_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 0.0 );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, ColVectorClone )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, Comm, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );

    Teuchos::RCP<const Teuchos::Comm<int> > copy_comm = MT::getComm( *A );

    TEST_EQUALITY( comm->getRank(), copy_comm->getRank() );
    TEST_EQUALITY( comm->getSize(), copy_comm->getSize() );
}

UNIT_TEST_INSTANTIATION( MatrixTraits, Comm )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, GlobalNumRows, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    TEST_EQUALITY( MT::getGlobalNumRows( *A ), global_num_rows );
}

UNIT_TEST_INSTANTIATION( MatrixTraits, GlobalNumRows )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, LocalNumRows, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    TEST_EQUALITY( MT::getLocalNumRows( *A ), local_num_rows );
}

UNIT_TEST_INSTANTIATION( MatrixTraits, LocalNumRows )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, GlobalMaxEntries, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    TEST_EQUALITY( MT::getGlobalMaxNumRowEntries( *A ), 1 );
}

UNIT_TEST_INSTANTIATION( MatrixTraits, GlobalMaxEntries )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, l2g_row, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    int offset = comm->getRank() * local_num_rows;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_EQUALITY( MT::getGlobalRow( *A, i ), i + offset );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, l2g_row )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, g2l_row, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    int offset = comm->getRank() * local_num_rows;
    for ( int i = offset; i < local_num_rows+offset; ++i )
    {
	TEST_EQUALITY( MT::getLocalRow( *A, i ), i );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, g2l_row )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, l2g_col, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    int offset = comm->getRank() * local_num_rows;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_EQUALITY( MT::getGlobalCol( *A, i ), i + offset );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, l2g_col )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, g2l_col, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    int offset = comm->getRank() * local_num_rows;
    for ( int i = offset; i < local_num_rows+offset; ++i )
    {
	TEST_EQUALITY( MT::getLocalCol( *A, i ), i );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, g2l_col )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, GlobalRowRanks, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );

    Teuchos::Array<GO> rows( global_num_rows );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	rows[i] = i;
    }

    Teuchos::Array<int> ranks( global_num_rows );
    MT::getGlobalRowRanks( *A, rows(), ranks() );

    for ( int i = 0; i < global_num_rows; ++i )
    {
	for ( int n = 0; n < comm_size; ++n )
	{
	    if ( i >= local_num_rows*n && i < local_num_rows*(n+1) )
	    {
		TEST_EQUALITY( ranks[i], n );
	    }
	}
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, GlobalRowRanks )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, is_l_row, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_ASSERT( MT::isLocalRow( *A, i ) );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, is_l_row )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, is_g_row, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    int offset = comm->getRank() * local_num_rows;
    for ( int i = offset; i < local_num_rows+offset; ++i )
    {
	TEST_ASSERT( MT::isGlobalRow( *A, i ) );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, is_g_row )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, is_l_col, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_ASSERT( MT::isLocalCol( *A, i ) );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, is_l_col )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, is_g_col, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    int offset = comm->getRank() * local_num_rows;
    for ( int i = offset; i < local_num_rows+offset; ++i )
    {
	TEST_ASSERT( MT::isGlobalCol( *A, i ) );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, is_g_col )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, g_row_copy, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    Teuchos::Array<GO> view_columns(1);
    Teuchos::Array<Scalar> view_values(1);
    std::size_t num_entries;
    int offset = comm->getRank() * local_num_rows;
    for ( int i = offset; i < local_num_rows+offset; ++i )
    {
	MT::getGlobalRowCopy( *A, i, view_columns(), view_values(), num_entries );
	TEST_EQUALITY( num_entries, 1 );
	TEST_EQUALITY( view_columns[0], i );
	TEST_EQUALITY( view_values[0], comm_size );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, g_row_copy )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, l_row_copy, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    std::size_t num_entries;
    Teuchos::Array<LO> view_columns(1);
    Teuchos::Array<Scalar> view_values(1);
    for ( int i = 0; i < local_num_rows; ++i )
    {
	MT::getLocalRowCopy( *A, i, view_columns(), view_values(), num_entries );
	TEST_EQUALITY( num_entries, 1 );
	TEST_EQUALITY( view_columns[0], i );
	TEST_EQUALITY( view_values[0], comm_size );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, l_row_copy )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, diag_copy, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    Teuchos::RCP<VectorType> X = MT::cloneVectorFromMatrixRows( *A );
    MT::getLocalDiagCopy( *A, *X );

    Teuchos::ArrayRCP<const Scalar> X_view = VT::view( *X );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator view_iterator;
    for ( view_iterator = X_view.begin();
	  view_iterator != X_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, comm_size );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, diag_copy )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, apply, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    Teuchos::RCP<VectorType> X = MT::cloneVectorFromMatrixRows( *A );
    Scalar x_fill = 2;
    VT::putScalar( *X, x_fill );
    Teuchos::RCP<VectorType> Y = VT::clone( *X );
    MT::apply( *A, *X, *Y );

    Teuchos::ArrayRCP<const Scalar> Y_view = VT::view( *Y );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator view_iterator;
    for ( view_iterator = Y_view.begin();
	  view_iterator != Y_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, comm_size*x_fill );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, apply )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, transpose, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 2 );
    Teuchos::Array<Scalar> values( 2 );
    for ( int i = 0; i < global_num_rows-1; ++i )
    {
	global_columns[0] = i;
	global_columns[1] = i+1;
	values[0] = 1;
	values[1] = 2;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    Teuchos::RCP<MatrixType> B = MT::copyTranspose( *A );

    std::size_t num_entries;
    Teuchos::Array<LO> view_columns(2);
    Teuchos::Array<Scalar> view_values(2);
    for ( int i = 1; i < local_num_rows-1; ++i )
    {
	MT::getLocalRowCopy( *B, i, view_columns, view_values, num_entries );
	TEST_EQUALITY( num_entries, 2 );
	TEST_EQUALITY( view_columns[0], i-1 );
	TEST_EQUALITY( view_columns[1], i );
	TEST_EQUALITY( view_values[0], 2*comm_size );
	TEST_EQUALITY( view_values[1], 1*comm_size );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, transpose )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, copy_neighbor, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( global_num_rows );
    Teuchos::Array<Scalar> values( global_num_rows, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[i] = i;
    }
    for ( int i = 0; i < global_num_rows; ++i )
    {
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    for ( int i = 0; i < 5; ++i )
    {
	Teuchos::RCP<MatrixType> B =  MT::copyNearestNeighbors( *A, i );

	int local_num_neighbor = 0;
	if ( i > 0 )
	{
	    local_num_neighbor = global_num_rows - local_num_rows;
	}

	TEST_EQUALITY( local_num_neighbor, MT::getLocalNumRows( *B ) );

	std::size_t num_entries;
	Teuchos::Array<LO> view_columns( global_num_rows );
	Teuchos::Array<Scalar> view_values( global_num_rows );
	for ( int j = 0; j < local_num_neighbor; ++j )
	{
	    for ( int k = comm_rank*local_num_rows; 
		  k < (comm_rank+1)*local_num_rows; ++k )
	    {
		TEST_INEQUALITY( MT::getGlobalRow( *B, j ), k );
	    }

	    MT::getLocalRowCopy( *B, j, view_columns, view_values, num_entries );
	    TEST_EQUALITY( num_entries, Teuchos::as<std::size_t>(global_num_rows) );

	    for ( int n = 0; n < global_num_rows; ++n )
	    {
	    	TEST_EQUALITY( view_columns[n], n );
	    	TEST_EQUALITY( view_values[n], comm_size );

	    }
	}
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, copy_neighbor )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, multiply, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 2 );
    Teuchos::Array<Scalar> values( 2 );
    for ( int i = 0; i < global_num_rows-1; ++i )
    {
	global_columns[0] = i;
	global_columns[1] = i+1;
	values[0] = 1.0/comm_size;
	values[1] = 2.0/comm_size;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    Teuchos::RCP<MatrixType> B = A;
    Teuchos::RCP<MatrixType> C = MT::copyTranspose( *B );
    Teuchos::RCP<MatrixType> D = MT::clone( *A );
    MT::multiply( B, C, D, false );

    std::size_t num_entries;
    Teuchos::Array<LO> view_columns(3);
    Teuchos::Array<Scalar> view_values(3);
    for ( int i = 1; i < local_num_rows-2; ++i )
    {
	MT::getLocalRowCopy( *D, i, view_columns, view_values, num_entries );
	TEST_EQUALITY( num_entries, 3 );
	TEST_EQUALITY( view_columns[0], i-1 );
	TEST_EQUALITY( view_columns[1], i );
	TEST_EQUALITY( view_columns[2], i+1 );
	TEST_EQUALITY( view_values[0], 2 );
	TEST_EQUALITY( view_values[1], 5 );
	TEST_EQUALITY( view_values[2], 2 );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, multiply )

//---------------------------------------------------------------------------//
// end tstTpetraCrsMatrix.cpp
//---------------------------------------------------------------------------//

