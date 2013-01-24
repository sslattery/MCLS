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
// end tstEpetraHelpers.cpp
//---------------------------------------------------------------------------//

