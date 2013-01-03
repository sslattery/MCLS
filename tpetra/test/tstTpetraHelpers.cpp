//---------------------------------------------------------------------------//
/*!
 * \file tstTpetraHelpers.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra helper function tests.
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

#include <MCLS_TpetraHelpers.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_BlockMap.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_VbrMatrix.hpp>

//---------------------------------------------------------------------------//
// Instantiation macro. 
// 
// These types are those enabled by Tpetra under explicit instantiation.
//---------------------------------------------------------------------------//
#define UNIT_TEST_INSTANTIATION( type, name )			           \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( type, name, int, int, int )      \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( type, name, int, int, long )     \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( type, name, int, int, double )   \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( type, name, int, long, int )     \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( type, name, int, long, long )    \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( type, name, int, long, double )

//---------------------------------------------------------------------------//
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( TpetraHelpers, CrsOffProcCols, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> OperatorType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_rank = comm->getRank();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    for ( int num_overlap = 0; num_overlap < 4; ++num_overlap )
    {
	Teuchos::RCP<OperatorType> A = 
	    Tpetra::createCrsMatrix<Scalar,LO,GO>( map );

	Teuchos::Array<GO> global_columns( 2*num_overlap+1 );
	Teuchos::Array<Scalar> values( 2*num_overlap+1, 1 );
	for ( int i = num_overlap; i < global_num_rows-num_overlap; ++i )
	{
	    for ( int j = 0; j < 2*num_overlap+1; ++j )
	    {
		global_columns[j] = i+j-num_overlap;
	    }
	    A->insertGlobalValues( i, global_columns(), values() );
	}
	A->fillComplete();

	Teuchos::Array<GO> off_proc_cols = 
	    MCLS::TpetraMatrixHelpers<OperatorType>::getOffProcRowsAsCols( *A );

	if ( comm_size == 1 )
	{
	    TEST_EQUALITY( off_proc_cols.size(), 0 );
	}
	else if ( comm_rank == 0 )
	{
	    TEST_EQUALITY( off_proc_cols.size(), num_overlap );

	    GO val = local_num_rows;
	    for ( int i = 0; i < num_overlap; ++i, ++val )
	    {
		TEST_EQUALITY( off_proc_cols[i], val );
	    }
	}
	else if ( comm_rank == comm_size-1 )
	{
	    TEST_EQUALITY( off_proc_cols.size(), num_overlap );

	    GO val = comm_rank*local_num_rows - num_overlap;
	    for ( int i = 0; i < num_overlap; ++i, ++val )
	    {
		TEST_EQUALITY( off_proc_cols[i], val );
	    }
	}
	else
	{
	    TEST_EQUALITY( off_proc_cols.size(), 2*num_overlap );

	    GO val = comm_rank*local_num_rows - num_overlap;
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

UNIT_TEST_INSTANTIATION( TpetraHelpers, CrsOffProcCols )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( TpetraHelpers, VbrOffProcCols, LO, GO, Scalar )
{
    typedef Tpetra::VbrMatrix<Scalar,LO,GO> OperatorType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_rank = comm->getRank();
    int comm_size = comm->getSize();

    int block_size = 1;
    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::BlockMap<LO,GO> > map = Teuchos::rcp(
	new Tpetra::BlockMap<LO,GO>( global_num_rows, block_size, 0, comm ) );

    for ( int num_overlap = 0; num_overlap < 4; ++num_overlap )
    {
    	Teuchos::RCP<OperatorType> A = Teuchos::rcp(
    	    new Tpetra::VbrMatrix<Scalar,LO,GO>( map, 2*num_overlap+1 ) );
	A->putScalar( 0 );

    	// for ( int i = 0; i < global_num_rows; ++i )
    	// {
	//     Teuchos::SerialDenseMatrix<LO,Scalar> one(1,1);
	//     one( 0, 0 ) = 1;
    	//     A->setGlobalBlockEntry( i, i, one );
    	// }

    	Teuchos::SerialDenseMatrix<LO,Scalar> block( 1, 2*num_overlap+1 );
    	for ( int j = 0; j < 2*num_overlap+1; ++j )
    	{
    	    block( 0, j ) = 1;
    	}

    	for ( int i = 0; i < global_num_rows; ++i )
    	{
    	    A->setGlobalBlockEntry( i, i, block );
    	}
    	A->fillComplete();

    	Teuchos::Array<GO> off_proc_cols = 
    	    MCLS::TpetraMatrixHelpers<OperatorType>::getOffProcRowsAsCols( *A );
	std::cout << "COL SIZE " << off_proc_cols.size() << std::endl;
    	if ( comm_size == 1 )
    	{
    	    TEST_EQUALITY( off_proc_cols.size(), 0 );
    	}
    	else if ( comm_rank == 0 )
    	{
    	    TEST_EQUALITY( off_proc_cols.size(), num_overlap );

    	    GO val = local_num_rows;
    	    for ( int i = 0; i < num_overlap; ++i, ++val )
    	    {
    		TEST_EQUALITY( off_proc_cols[i], val );
    	    }
    	}
    	else if ( comm_rank == comm_size-1 )
    	{
    	    TEST_EQUALITY( off_proc_cols.size(), num_overlap );

    	    GO val = comm_rank*local_num_rows - num_overlap;
    	    for ( int i = 0; i < num_overlap; ++i, ++val )
    	    {
    		TEST_EQUALITY( off_proc_cols[i], val );
    	    }
    	}
    	else
    	{
    	    TEST_EQUALITY( off_proc_cols.size(), 2*num_overlap );

    	    GO val = comm_rank*local_num_rows - num_overlap;
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

UNIT_TEST_INSTANTIATION( TpetraHelpers, VbrOffProcCols )

//---------------------------------------------------------------------------//
// end tstTpetraHelpers.cpp
//---------------------------------------------------------------------------//

