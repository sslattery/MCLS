//---------------------------------------------------------------------------//
/*!
 * \file tstTpetraAdjointTally.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra AdjointTally tests.
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

#include <MCLS_AdjointTally.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_TpetraAdapter.hpp>
#include <MCLS_History.hpp>

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
TEUCHOS_UNIT_TEST( AdjointTally, Typedefs )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef MCLS::History<Scalar,int> HistoryType;
    typedef typename TallyType::HistoryType history_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<HistoryType, history_type>::value)
	== true, true );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( AdjointTally, TallyHistory )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::History<double,int> HistoryType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map_a = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );
    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map_a ) );

    Teuchos::Array<int> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::RCP<Epetra_Map> map_b = Teuchos::rcp(
		new Epetra_Map( -1, 
				Teuchos::as<int>(inverse_rows.size()),
				inverse_rows.getRawPtr(),
				0,
				*epetra_comm ) );
    Teuchos::RCP<VectorType> B = Teuchos::rcp( new Epetra_Vector( *map_b ) );

    MCLS::AdjointTally<VectorType> tally( A, B );
    double a_val = 2;
    double b_val = 3;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	int state = i + local_num_rows*comm_rank;
	HistoryType history( state, a_val );
	history.live();
	tally.tallyHistory( history );

	int inverse_state = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
	history = HistoryType( inverse_state, b_val );
	history.live();
	tally.tallyHistory( history );
    }
    
    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const double>::const_iterator a_view_iterator;
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
	if ( comm_size == 1 )
	{
	    TEST_EQUALITY( *a_view_iterator, a_val + b_val );
	}
	else
	{
	    TEST_EQUALITY( *a_view_iterator, a_val );
	}
    }

    Teuchos::ArrayRCP<const double> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const double>::const_iterator b_view_iterator;
    for ( b_view_iterator = B_view.begin();
	  b_view_iterator != B_view.end();
	  ++b_view_iterator )
    {
	if ( comm_size == 1 )
	{
	    TEST_EQUALITY( *b_view_iterator, 0 );
	}
	else
	{
	    TEST_EQUALITY( *b_view_iterator, b_val );
	}
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( AdjointTally, Combine )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::History<double,int> HistoryType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map_a = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );
    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map_a ) );

    Teuchos::Array<int> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::RCP<Epetra_Map> map_b = Teuchos::rcp(
		new Epetra_Map( -1, 
				Teuchos::as<int>(inverse_rows.size()),
				inverse_rows.getRawPtr(),
				0,
				*epetra_comm ) );
    Teuchos::RCP<VectorType> B = Teuchos::rcp( new Epetra_Vector( *map_b ) );

    MCLS::AdjointTally<VectorType> tally( A, B );

    double a_val = 2;
    double b_val = 3;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	int state = i + local_num_rows*comm_rank;
	HistoryType history( state, a_val );
	history.live();
	tally.tallyHistory( history );

	int inverse_state = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
	history = HistoryType( inverse_state, b_val );
	history.live();
	tally.tallyHistory( history );
    }

    tally.combineTallies();

    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const double>::const_iterator a_view_iterator;
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
	TEST_EQUALITY( *a_view_iterator, a_val + b_val );
    }

    Teuchos::ArrayRCP<const double> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const double>::const_iterator b_view_iterator;
    for ( b_view_iterator = B_view.begin();
	  b_view_iterator != B_view.end();
	  ++b_view_iterator )
    {
	if ( comm_size == 1 )
	{
	    TEST_EQUALITY( *b_view_iterator, 0 );
	}
	else
	{
	    TEST_EQUALITY( *b_view_iterator, b_val );
	}
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( AdjointTally, Normalize )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::History<double,int> HistoryType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map_a = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );
    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map_a ) );

    Teuchos::Array<int> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::RCP<Epetra_Map> map_b = Teuchos::rcp(
		new Epetra_Map( -1, 
				Teuchos::as<int>(inverse_rows.size()),
				inverse_rows.getRawPtr(),
				0,
				*epetra_comm ) );
    Teuchos::RCP<VectorType> B = Teuchos::rcp( new Epetra_Vector( *map_b ) );

    MCLS::AdjointTally<VectorType> tally( A, B );
    double a_val = 2;
    double b_val = 3;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	int state = i + local_num_rows*comm_rank;
	HistoryType history( state, a_val );
	history.live();
	tally.tallyHistory( history );

	int inverse_state = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
	history = HistoryType( inverse_state, b_val );
	history.live();
	tally.tallyHistory( history );
    }
    
    tally.combineTallies();
    int nh = 10;
    tally.normalize( nh );

    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const double>::const_iterator a_view_iterator;
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
	TEST_EQUALITY( *a_view_iterator, (a_val + b_val) / nh );
    }

    Teuchos::ArrayRCP<const double> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const double>::const_iterator b_view_iterator;
    for ( b_view_iterator = B_view.begin();
	  b_view_iterator != B_view.end();
	  ++b_view_iterator )
    {
	if ( comm_size == 1 )
	{
	    TEST_EQUALITY( *b_view_iterator, 0 );
	}
	else
	{
	    TEST_EQUALITY( *b_view_iterator, b_val );
	}
    }
}

//---------------------------------------------------------------------------//
// end tstEpetraAdjointTally.cpp
//---------------------------------------------------------------------------//

