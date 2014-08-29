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
 * \file tstEpetraHelpers.cpp
 * \author Stuart R. Slattery
 * \brief Epetra helper function tests.
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

#include <MCLS_EpetraHelpers.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

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
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( EpetraHelpers, OffProcCols )
{
    typedef Epetra_RowMatrix MatrixType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();
    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;

    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    for ( int num_overlap = 0; num_overlap < 4; ++num_overlap )
    {
	Teuchos::RCP<Epetra_CrsMatrix> A = 
	    Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

	Teuchos::Array<int> global_columns( 2*num_overlap+1 );
	Teuchos::Array<double> values( 2*num_overlap+1, 1 );
	for ( int i = num_overlap; i < global_num_rows-num_overlap; ++i )
	{
	    for ( int j = 0; j < 2*num_overlap+1; ++j )
	    {
		global_columns[j] = i+j-num_overlap;
	    }
	    A->InsertGlobalValues( i, global_columns.size(),
				   &values[0], &global_columns[0] );
	}
	A->FillComplete();

	Teuchos::Array<int> off_proc_cols = 
	    MCLS::EpetraMatrixHelpers<MatrixType>::getOffProcColsAsRows( *A );

	if ( comm_size == 1 )
	{
	    TEST_EQUALITY( off_proc_cols.size(), 0 );
	}
	else if ( comm_rank == 0 )
	{
	    TEST_EQUALITY( off_proc_cols.size(), num_overlap );

	    int val = local_num_rows;
	    for ( int i = 0; i < num_overlap; ++i, ++val )
	    {
		TEST_EQUALITY( off_proc_cols[i], val );
	    }
	}
	else if ( comm_rank == comm_size-1 )
	{
	    TEST_EQUALITY( off_proc_cols.size(), num_overlap );

	    int val = comm_rank*local_num_rows - num_overlap;
	    for ( int i = 0; i < num_overlap; ++i, ++val )
	    {
		TEST_EQUALITY( off_proc_cols[i], val );
	    }
	}
	else
	{
	    TEST_EQUALITY( off_proc_cols.size(), 2*num_overlap );

	    int val = comm_rank*local_num_rows - num_overlap;
	    for ( int i = 0; i < num_overlap; ++i, ++val )
	    {
		TEST_EQUALITY( off_proc_cols[i], val );
	    }

	    val = (comm_rank+1)*local_num_rows;
	    for ( int i = 0; i < num_overlap; ++i, ++val )
	    {
		TEST_EQUALITY( off_proc_cols[i], val );
	    }
	}

	comm->barrier();
    }
}

//---------------------------------------------------------------------------//
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( EpetraHelpers, MatrixFilter )
{
    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    // Build the linear operator and solution vector.
    Teuchos::RCP<Epetra_CrsMatrix> A = 	
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
    Teuchos::Array<int> first_columns( 1, 0 );
    Teuchos::Array<double> first_values( 1, 2.0 );
    A->InsertGlobalValues( 0, first_columns().size(), 
                           &first_values[0], &first_columns[0] );
    Teuchos::Array<int> global_columns( 2 );
    Teuchos::Array<double> values( 2 );
    for ( int i = 1; i < global_num_rows; ++i )
    {
	global_columns[0] = i-1;
	global_columns[1] = i;
	values[0] = 2;
	values[1] = 3;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );
    }
    A->FillComplete();

    Teuchos::RCP<Epetra_CrsMatrix> filter_A =
	MCLS::EpetraMatrixHelpers<Epetra_RowMatrix>::filter( *A, 0.0 );

    TEST_ASSERT( filter_A->Filled() );
    TEST_EQUALITY( A->MaxNumEntries(), filter_A->MaxNumEntries() );
    TEST_EQUALITY( A->NumMyRows(), filter_A->NumMyRows() );
    TEST_EQUALITY( A->NumMyCols(), filter_A->NumMyCols() );
    int max_size = A->MaxNumEntries();
    Teuchos::Array<double> A_values( max_size );
    Teuchos::Array<int> A_indices( max_size );
    Teuchos::Array<double> filter_A_values( max_size );
    Teuchos::Array<int> filter_A_indices( max_size );
    int error = 0;
    int num_entries_A = 0;
    int num_entries_filter_A = 0;
    for ( int i = 0; i < A->NumMyRows(); ++i )
    {
	TEST_EQUALITY( A->GRID(i), filter_A->GRID(i) );

	error = A->ExtractMyRowCopy( i, max_size, num_entries_A,
				     A_values.getRawPtr(),
				     A_indices.getRawPtr() );
	TEST_EQUALITY( 0, error );

	error = filter_A->ExtractMyRowCopy( i, max_size, num_entries_filter_A,
					    filter_A_values.getRawPtr(),
					    filter_A_indices.getRawPtr() );
	TEST_EQUALITY( 0, error );

	TEST_EQUALITY( num_entries_A, num_entries_filter_A );

	for ( int j = 0; j < num_entries_A; ++j )
	{
	    TEST_EQUALITY( A_values[j], filter_A_values[j] );
	    TEST_EQUALITY( A_indices[j], filter_A_indices[j] );
	    TEST_EQUALITY( A->GCID(A_indices[j]),
			   filter_A->GCID(filter_A_indices[j]) );
	}
    }
}

//---------------------------------------------------------------------------//
// end tstEpetraHelpers.cpp
//---------------------------------------------------------------------------//

