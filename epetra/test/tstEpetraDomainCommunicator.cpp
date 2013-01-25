//---------------------------------------------------------------------------//
/*!
 * \file tstEpetraDomainCommunicator.cpp
 * \author Stuart R. Slattery
 * \brief Epetra AdjointDomain tests.
 */
//---------------------------------------------------------------------------//

#include <stack>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <string>
#include <cassert>

#include <MCLS_DomainCommunicator.hpp>
#include <MCLS_DomainTransporter.hpp>
#include <MCLS_AdjointDomain.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_EpetraAdapter.hpp>
#include <MCLS_History.hpp>
#include <MCLS_AdjointTally.hpp>
#include <MCLS_Events.hpp>
#include <MCLS_RNGControl.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Comm.h>
#include <Epetra_SerialComm.h>

#ifdef HAVE_MPI
#include <Epetra_MpiComm.h>
#include <Teuchos_DefaultMpiComm.hpp>
#endif

//---------------------------------------------------------------------------//
// Helper functions.
//---------------------------------------------------------------------------//

Teuchos::RCP<const Epetra_Comm> getEpetraComm()
{
#ifdef HAVE_MPI
    return Teuchos::rcp( new Epetra_MpiComm(MPI_COMM_WORLD) );
#else
    return Teuchos::rcp( new Epetra_SerialComm() );
#endif
}

//---------------------------------------------------------------------------//
Teuchos::RCP<const Teuchos::Comm<int> > getTeuchosCommFromEpetra(
    const Teuchos::RCP<const Epetra_Comm>& epetra_comm )
{
    Teuchos::RCP<const Teuchos::MpiComm<int> > teuchos_comm;

#ifdef HAVE_MPI
	Teuchos::RCP<const Epetra_MpiComm> mpi_epetra_comm =
	    Teuchos::rcp_dynamic_cast<const Epetra_MpiComm>( epetra_comm );

	Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm> >
	    raw_mpi_comm = Teuchos::opaqueWrapper( mpi_epetra_comm->Comm() );

	teuchos_comm =
	    Teuchos::rcp( new Teuchos::MpiComm<int>( raw_mpi_comm ) );
#else
	teuchos_comm = Teuchos::DefaultComm<int>::getComm();
#endif

	return teuchos_comm;
}

//---------------------------------------------------------------------------//
// RNG setup.
//---------------------------------------------------------------------------//
MCLS::RNGControl control( 2394723 );

//---------------------------------------------------------------------------//
// Helper functions.
//---------------------------------------------------------------------------//
Teuchos::RCP<MCLS::History<int> > makeHistory( 
    int state, double weight, int streamid )
{
    Teuchos::RCP<MCLS::History<int> > history = Teuchos::rcp(
	new MCLS::History<int>( state, weight ) );
    history->setRNG( control.rng(streamid) );
    history->setEvent( MCLS::BOUNDARY );
    return history;
}

//---------------------------------------------------------------------------//
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( DomainCommunicator, Typedefs )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;
    typedef MCLS::History<int> HistoryType;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;

    typedef MCLS::DomainTransporter<DomainType> TransportType;
    typedef TransportType::HistoryType history_type;
    typedef TransportType::BankType bank_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<HistoryType, history_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<bank_type, 
	 std::stack<Teuchos::RCP<HistoryType> > >::value)
	 == true, true );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( DomainCommunicator, Communicate )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::History<int> HistoryType;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;

    Teuchos::RCP<const Epetra_Comm> epetra_comm = getEpetraComm();
    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	getTeuchosCommFromEpetra( epetra_comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    // This test is parallel.
    if ( comm_size > 1 )
    {
	int local_num_rows = 10;
	int global_num_rows = local_num_rows*comm_size;
	Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	    new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

	// Build the linear operator and solution vector.
	Teuchos::RCP<Epetra_CrsMatrix> A = 	
	    Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
	Teuchos::Array<int> global_columns( 1 );
	Teuchos::Array<double> values( 1 );
	for ( int i = 1; i < global_num_rows; ++i )
	{
	    global_columns[0] = i-1;
	    values[0] = -0.5;
	    A->InsertGlobalValues( i, global_columns.size(), 
				   &values[0], &global_columns[0] );
	}
	global_columns[0] = global_num_rows-1;
	values[0] = -0.5;
	A->InsertGlobalValues( global_num_rows-1, global_columns.size(),
			       &values[0], &global_columns[0] );
	A->FillComplete();

	Teuchos::RCP<MatrixType> B = A;
	Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *B );

	// Build the adjoint domain.
	Teuchos::ParameterList plist;
	plist.set<int>( "Overlap Size", 0 );
	Teuchos::RCP<DomainType> domain = 
	    Teuchos::rcp( new DomainType( B, x, plist ) );

	// History setup.
	HistoryType::setByteSize( control.getSize() );

	// Build the domain communicator.
	MCLS::DomainCommunicator<DomainType>::BankType bank;
	int buffer_size = 3;
	plist.set<int>( "History Buffer Size", buffer_size );

	MCLS::DomainCommunicator<DomainType> communicator( 
	    domain, MT::getComm(*B), plist );

	// Test initialization.
	TEST_EQUALITY( Teuchos::as<int>(communicator.maxBufferSize()), buffer_size );
	TEST_ASSERT( !communicator.sendStatus() );
	TEST_ASSERT( !communicator.receiveStatus() );

	// Post receives.
	communicator.post();
	if ( comm_rank == 0 )
	{
	    TEST_ASSERT( !communicator.receiveStatus() );
	}
	else
	{
	    TEST_ASSERT( communicator.receiveStatus() );
	}

	// End communication.
	communicator.end();
	TEST_ASSERT( !communicator.receiveStatus() );

	// Post new receives.
	communicator.post();
	if ( comm_rank == 0 )
	{
	    TEST_ASSERT( !communicator.receiveStatus() );
	}
	else
	{
	    TEST_ASSERT( communicator.receiveStatus() );
	}
	TEST_EQUALITY( communicator.sendBufferSize(), 0 );

	// Flush with zero histories.
	TEST_EQUALITY( communicator.flush(), 0 );
	if ( comm_rank == 0 )
	{
	    TEST_ASSERT( !communicator.receiveStatus() );
	}
	else
	{
	    TEST_ASSERT( communicator.receiveStatus() );
	}
	TEST_ASSERT( !communicator.sendStatus() );

	// Receive empty flushed buffers.
	int zero_histories = communicator.wait( bank );
	TEST_EQUALITY( zero_histories, 0 );
	TEST_ASSERT( !communicator.receiveStatus() );
	TEST_ASSERT( bank.empty() );

	// Repost receives.
	communicator.post();
	if ( comm_rank == 0 )
	{
	    TEST_ASSERT( !communicator.receiveStatus() );
	}
	else
	{
	    TEST_ASSERT( communicator.receiveStatus() );
	}

	// Proc 0 will send to proc 1.
	if ( comm_rank == 0 )
	{
	    TEST_ASSERT( !domain->isLocalState(10) );

	    Teuchos::RCP<HistoryType> h1 = 
		makeHistory( 10, 1.1, comm_rank*4 + 1 );
	    const MCLS::DomainCommunicator<DomainType>::Result
		r1 = communicator.communicate( h1 );
	    TEST_ASSERT( !r1.sent );
	    TEST_EQUALITY( communicator.sendBufferSize(), 1 );

	    Teuchos::RCP<HistoryType> h2 = 
		makeHistory( 10, 2.1, comm_rank*4 + 2 );
	    const MCLS::DomainCommunicator<DomainType>::Result
		r2 = communicator.communicate( h2 );
	    TEST_ASSERT( !r2.sent );
	    TEST_EQUALITY( communicator.sendBufferSize(), 2 );

	    Teuchos::RCP<HistoryType> h3 = 
		makeHistory( 10, 3.1, comm_rank*4 + 3 );
	    const MCLS::DomainCommunicator<DomainType>::Result
		r3 = communicator.communicate( h3 );
	    TEST_ASSERT( r3.sent );
	    TEST_EQUALITY( r3.destination, 1 );
	    TEST_EQUALITY( communicator.sendBufferSize(), 0 );
	}

	// Proc comm_rank send to proc comm_rank+1 and receive from proc
	// comm_rank-1. 
	else if ( comm_rank < comm_size - 1 )
	{
	    // Send to proc comm_rank+1.
	    TEST_ASSERT( !domain->isLocalState((comm_rank+1)*10) );

	    Teuchos::RCP<HistoryType> h1 = 
		makeHistory( (comm_rank+1)*10, 1.1, comm_rank*4 + 1 );
	    const MCLS::DomainCommunicator<DomainType>::Result
		r1 = communicator.communicate( h1 );
	    TEST_ASSERT( !r1.sent );
	    TEST_EQUALITY( communicator.sendBufferSize(), 1 );

	    Teuchos::RCP<HistoryType> h2 = 
		makeHistory( (comm_rank+1)*10, 2.1, comm_rank*4 + 2 );
	    const MCLS::DomainCommunicator<DomainType>::Result
		r2 = communicator.communicate( h2 );
	    TEST_ASSERT( !r2.sent );
	    TEST_EQUALITY( communicator.sendBufferSize(), 2 );

	    Teuchos::RCP<HistoryType> h3 = 
		makeHistory( (comm_rank+1)*10, 3.1, comm_rank*4 + 3 );
	    const MCLS::DomainCommunicator<DomainType>::Result
		r3 = communicator.communicate( h3 );
	    TEST_ASSERT( r3.sent );
	    TEST_EQUALITY( r3.destination, comm_rank+1 );
	    TEST_EQUALITY( communicator.sendBufferSize(), 0 );

	    // Receive from proc comm_rank-1.
	    while ( bank.empty() )
	    {
		communicator.checkAndPost( bank );
	    }

	    TEST_EQUALITY( bank.size(), 3 );

	    Teuchos::RCP<HistoryType> rp3 = bank.top();
	    bank.pop();
	    Teuchos::RCP<HistoryType> rp2 = bank.top();
	    bank.pop();
	    Teuchos::RCP<HistoryType> rp1 = bank.top();
	    bank.pop();
	    TEST_ASSERT( bank.empty() );

	    TEST_EQUALITY( rp3->state(), comm_rank*10 );
	    TEST_EQUALITY( rp3->weight(), 3.1 );
	    TEST_EQUALITY( rp2->state(), comm_rank*10 );
	    TEST_EQUALITY( rp2->weight(), 2.1 );
	    TEST_EQUALITY( rp1->state(), comm_rank*10 );
	    TEST_EQUALITY( rp1->weight(), 1.1 );

	    TEST_ASSERT( communicator.receiveStatus() );
	}

	// The last proc just receives.
	else
	{
	    // Check and post until receive from proc comm_rank-1
	    while ( bank.empty() )
	    {
		communicator.checkAndPost( bank );
	    }
	    TEST_ASSERT( communicator.receiveStatus() );
	    TEST_EQUALITY( bank.size(), 3 );

	    Teuchos::RCP<HistoryType> rp3 = bank.top();
	    bank.pop();
	    Teuchos::RCP<HistoryType> rp2 = bank.top();
	    bank.pop();
	    Teuchos::RCP<HistoryType> rp1 = bank.top();
	    bank.pop();
	    TEST_ASSERT( bank.empty() );

	    TEST_EQUALITY( rp3->state(), comm_rank*10 );
	    TEST_EQUALITY( rp3->weight(), 3.1 );
	    TEST_EQUALITY( rp2->state(), comm_rank*10 );
	    TEST_EQUALITY( rp2->weight(), 2.1 );
	    TEST_EQUALITY( rp1->state(), comm_rank*10 );
	    TEST_EQUALITY( rp1->weight(), 1.1 );
	}

	// End communication.
	communicator.end();
	TEST_ASSERT( !communicator.receiveStatus() );
    }

    // Barrier before exiting to make sure memory deallocation happened
    // correctly. 
    comm->barrier();
}

//---------------------------------------------------------------------------//
// end tstEpetraDomainCommunicator.cpp
//---------------------------------------------------------------------------//

