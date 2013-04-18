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
 * \file tstEpetraRowMatrix.cpp
 * \author Stuart R. Slattery
 * \brief Epetra_RowMatrix adapter tests.
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
#include <MCLS_EpetraAdapter.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>

#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Comm.h>
#include <Epetra_SerialComm.h>

#ifdef HAVE_MPI
#include <Epetra_MpiComm.h>
#endif

//---------------------------------------------------------------------------//
// Helper functions.
//---------------------------------------------------------------------------//

Teuchos::RCP<Epetra_Comm> getEpetraComm( 
    const Teuchos::RCP<const Teuchos::Comm<int> >& comm )
{
#ifdef HAVE_MPI
    Teuchos::RCP< const Teuchos::MpiComm<int> > mpi_comm = 
	Teuchos::rcp_dynamic_cast< const Teuchos::MpiComm<int> >( comm );
    Teuchos::RCP< const Teuchos::OpaqueWrapper<MPI_Comm> > opaque_comm = 
	mpi_comm->getRawMpiComm();
    return Teuchos::rcp( new Epetra_MpiComm( (*opaque_comm)() ) );
#else
    return Teuchos::rcp( new Epetra_SerialComm() );
#endif
}

//---------------------------------------------------------------------------//
// Test templates.
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, Typedefs )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<double, scalar_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<int, local_ordinal_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<int, global_ordinal_type>::value)
	== true, true );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, RowVectorClone )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();
    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;

    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::RCP<MatrixType> B = A;
    Teuchos::RCP<VectorType> X = MT::cloneVectorFromMatrixRows( *B );

    TEST_ASSERT( A->RowMap().SameAs( X->Map() ) );

    Teuchos::ArrayRCP<const double> X_view = VT::view( *X );
    Teuchos::ArrayRCP<const double>::const_iterator view_iterator;
    for ( view_iterator = X_view.begin();
	  view_iterator != X_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 0.0 );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, ColVectorClone )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );

    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );
    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    Teuchos::RCP<VectorType> X = MT::cloneVectorFromMatrixCols( *B );

    TEST_ASSERT( A->ColMap().SameAs( X->Map() ) );

    Teuchos::ArrayRCP<const double> X_view = VT::view( *X );
    Teuchos::ArrayRCP<const double>::const_iterator view_iterator;
    for ( view_iterator = X_view.begin();
	  view_iterator != X_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 0.0 );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, Comm )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::RCP<MatrixType> B = A;
    Teuchos::RCP<const Teuchos::Comm<int> > copy_comm = MT::getComm( *B );

    TEST_EQUALITY( comm->getRank(), copy_comm->getRank() );
    TEST_EQUALITY( comm->getSize(), copy_comm->getSize() );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, GlobalNumRows )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    TEST_EQUALITY( MT::getGlobalNumRows( *B ), global_num_rows );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, LocalNumRows )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    TEST_EQUALITY( MT::getLocalNumRows( *B ), local_num_rows );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, GlobalMaxEntries )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    TEST_EQUALITY( MT::getGlobalMaxNumRowEntries( *B ), 1 );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, l2g_row )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    int offset = comm->getRank() * local_num_rows;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_EQUALITY( MT::getGlobalRow( *B, i ), i + offset );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, g2l_row )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    int offset = comm->getRank() * local_num_rows;
    for ( int i = offset; i < local_num_rows+offset; ++i )
    {
	TEST_EQUALITY( MT::getLocalRow( *B, i ), i );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, l2g_col )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    int offset = comm->getRank() * local_num_rows;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_EQUALITY( MT::getGlobalCol( *B, i ), i + offset );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, g2l_col )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    int offset = comm->getRank() * local_num_rows;
    for ( int i = offset; i < local_num_rows+offset; ++i )
    {
	TEST_EQUALITY( MT::getLocalCol( *B, i ), i );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, GlobalRowRanks )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();
    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;

    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> rows( global_num_rows );
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

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, is_l_row )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_ASSERT( MT::isLocalRow( *B, i ) );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, is_g_row )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    int offset = comm->getRank() * local_num_rows;
    for ( int i = offset; i < local_num_rows+offset; ++i )
    {
	TEST_ASSERT( MT::isGlobalRow( *B, i ) );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, is_l_col )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_ASSERT( MT::isLocalCol( *B, i ) );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, is_g_col )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    int offset = comm->getRank() * local_num_rows;
    for ( int i = offset; i < local_num_rows+offset; ++i )
    {
	TEST_ASSERT( MT::isGlobalCol( *B, i ) );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, g_row_copy )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    std::size_t num_entries;
    Teuchos::Array<int> view_columns(1);
    Teuchos::Array<double> view_values(1);
    int offset = comm->getRank() * local_num_rows;
    for ( int i = offset; i < local_num_rows+offset; ++i )
    {
	MT::getGlobalRowCopy( *B, i, view_columns(), view_values(), num_entries );
	TEST_EQUALITY( num_entries, 1 );
	TEST_EQUALITY( view_columns[0], i );
	TEST_EQUALITY( view_values[0], 1 );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, l_row_copy )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    std::size_t num_entries;
    Teuchos::Array<int> view_columns(1);
    Teuchos::Array<double> view_values(1);
    for ( int i = 0; i < local_num_rows; ++i )
    {
	MT::getLocalRowCopy( *B, i, view_columns(), view_values(), num_entries );
	TEST_EQUALITY( num_entries, 1 );
	TEST_EQUALITY( view_columns[0], i );
	TEST_EQUALITY( view_values[0], 1 );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, diag_copy )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    Teuchos::RCP<VectorType> X = MT::cloneVectorFromMatrixRows( *B );
    MT::getLocalDiagCopy( *B, *X );

    Teuchos::ArrayRCP<const double> X_view = VT::view( *X );
    Teuchos::ArrayRCP<const double>::const_iterator view_iterator;
    for ( view_iterator = X_view.begin();
	  view_iterator != X_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 1 );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, apply )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    Teuchos::RCP<VectorType> X = MT::cloneVectorFromMatrixRows( *B );
    double x_fill = 2;
    VT::putScalar( *X, x_fill );
    Teuchos::RCP<VectorType> Y = VT::clone( *X );
    MT::apply( *B, *X, *Y );

    Teuchos::ArrayRCP<const double> Y_view = VT::view( *Y );
    Teuchos::ArrayRCP<const double>::const_iterator view_iterator;
    for ( view_iterator = Y_view.begin();
	  view_iterator != Y_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, x_fill );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, transpose )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 2 );
    Teuchos::Array<double> values( 2 );
    for ( int i = 0; i < global_num_rows-1; ++i )
    {
	global_columns[0] = i;
	global_columns[1] = i+1;
	values[0] = 1;
	values[1] = 2;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    Teuchos::RCP<MatrixType> C = MT::copyTranspose( *B );

    std::size_t num_entries;
    Teuchos::Array<int> view_columns(2);
    Teuchos::Array<double> view_values(2);
    for ( int i = 1; i < local_num_rows-1; ++i )
    {
	MT::getLocalRowCopy( *C, i, view_columns, view_values, num_entries );
	TEST_EQUALITY( num_entries, 2 );
	TEST_EQUALITY( view_columns[0], i-1 );
	TEST_EQUALITY( view_columns[1], i );
	TEST_EQUALITY( view_values[0], 2 );
	TEST_EQUALITY( view_values[1], 1 );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, copy_neighbor )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( global_num_rows );
    Teuchos::Array<double> values( global_num_rows, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[i] = i;
    }
    for ( int i = 0; i < global_num_rows; ++i )
    {
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    for ( int i = 0; i < 5; ++i )
    {
	Teuchos::RCP<MatrixType> C = MT::copyNearestNeighbors( *B, i );

	int local_num_neighbor = 0;
	if ( i > 0 )
	{
	    local_num_neighbor = global_num_rows - local_num_rows;
	}

	TEST_EQUALITY( local_num_neighbor, MT::getLocalNumRows( *C ) );

	std::size_t num_entries;
	Teuchos::Array<int> view_columns( global_num_rows );
	Teuchos::Array<double> view_values( global_num_rows );
	for ( int j = 0; j < local_num_neighbor; ++j )
	{
	    for ( int k = comm_rank*local_num_rows; 
		  k < (comm_rank+1)*local_num_rows; ++k )
	    {
		TEST_INEQUALITY( MT::getGlobalRow( *C, j ), k );
	    }

	    MT::getLocalRowCopy( *C, j, view_columns, view_values, num_entries );
	    TEST_EQUALITY( num_entries, Teuchos::as<std::size_t>(global_num_rows) );

	    for ( int n = 0; n < global_num_rows; ++n )
	    {
	    	TEST_EQUALITY( view_columns[n], n );
	    	TEST_EQUALITY( view_values[n], 1 );
	    }
	}
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MatrixTraits, multiply )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MT::scalar_type scalar_type;
    typedef MT::local_ordinal_type local_ordinal_type;
    typedef MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<int> global_columns( 2 );
    Teuchos::Array<double> values( 2 );
    for ( int i = 0; i < global_num_rows-1; ++i )
    {
	global_columns[0] = i;
	global_columns[1] = i+1;
	values[0] = 1;
	values[1] = 2;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );

    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    Teuchos::RCP<MatrixType> C = MT::copyTranspose( *B );
    Teuchos::RCP<MatrixType> D = MT::clone( *A );
    MT::multiply( B, C, D, false );

    std::size_t num_entries;
    Teuchos::Array<int> view_columns(3);
    Teuchos::Array<double> view_values(3);
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

//---------------------------------------------------------------------------//
// end tstEpetraRowMatrix.cpp
//---------------------------------------------------------------------------//

