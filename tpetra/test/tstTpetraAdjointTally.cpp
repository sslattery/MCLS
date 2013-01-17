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

#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

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
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( AdjointTally, Typedefs, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef MCLS::History<Scalar,GO> HistoryType;
    typedef typename TallyType::HistoryType history_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<HistoryType, history_type>::value)
	== true, true );
}

UNIT_TEST_INSTANTIATION( AdjointTally, Typedefs )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( AdjointTally, TallyHistory, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::History<Scalar,GO> HistoryType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_a = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );
    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map_a );

    Teuchos::Array<GO> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_b = 
	Tpetra::createNonContigMap<LO,GO>( inverse_rows(), comm );
    Teuchos::RCP<VectorType> B = Tpetra::createVector<Scalar,LO,GO>( map_b );

    MCLS::AdjointTally<VectorType> tally( A, B );
    Scalar a_val = 2;
    Scalar b_val = 3;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	GO state = i + local_num_rows*comm_rank;
	HistoryType history( state, a_val );
	history.live();
	tally.tallyHistory( history );

	GO inverse_state = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
	history = HistoryType( inverse_state, b_val );
	history.live();
	tally.tallyHistory( history );
    }
    
    Teuchos::ArrayRCP<const Scalar> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator a_view_iterator;
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

    Teuchos::ArrayRCP<const Scalar> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator b_view_iterator;
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

UNIT_TEST_INSTANTIATION( AdjointTally, TallyHistory )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( AdjointTally, Combine, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::History<Scalar,GO> HistoryType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_a = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );
    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map_a );

    Teuchos::Array<GO> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_b = 
	Tpetra::createNonContigMap<LO,GO>( inverse_rows(), comm );
    Teuchos::RCP<VectorType> B = Tpetra::createVector<Scalar,LO,GO>( map_b );

    MCLS::AdjointTally<VectorType> tally( A, B );

    Scalar a_val = 2;
    Scalar b_val = 3;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	GO state = i + local_num_rows*comm_rank;
	HistoryType history( state, a_val );
	history.live();
	tally.tallyHistory( history );

	GO inverse_state = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
	history = HistoryType( inverse_state, b_val );
	history.live();
	tally.tallyHistory( history );
    }

    tally.combineTallies();

    Teuchos::ArrayRCP<const Scalar> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator a_view_iterator;
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
	if ( comm_size == 1 )
	{
	    TEST_EQUALITY( *a_view_iterator, 0 );
	}
	else
	{
	    TEST_EQUALITY( *a_view_iterator, a_val + b_val );
	}
    }

    Teuchos::ArrayRCP<const Scalar> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator b_view_iterator;
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

UNIT_TEST_INSTANTIATION( AdjointTally, Combine )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( AdjointTally, Normalize, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::History<Scalar,GO> HistoryType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_a = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );
    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map_a );

    Teuchos::Array<GO> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_b = 
	Tpetra::createNonContigMap<LO,GO>( inverse_rows(), comm );
    Teuchos::RCP<VectorType> B = Tpetra::createVector<Scalar,LO,GO>( map_b );

    MCLS::AdjointTally<VectorType> tally( A, B );
    Scalar a_val = 2;
    Scalar b_val = 3;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	GO state = i + local_num_rows*comm_rank;
	HistoryType history( state, a_val );
	history.live();
	tally.tallyHistory( history );

	GO inverse_state = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
	history = HistoryType( inverse_state, b_val );
	history.live();
	tally.tallyHistory( history );
    }
    
    tally.combineTallies();
    int nh = 10;
    tally.normalize( nh );

    Teuchos::ArrayRCP<const Scalar> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator a_view_iterator;
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
	if ( comm_size == 1 )
	{
	    TEST_EQUALITY( *a_view_iterator, 0 );
	}
	else
	{
	    TEST_EQUALITY( *a_view_iterator, (a_val + b_val) / nh );
	}
    }

    Teuchos::ArrayRCP<const Scalar> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator b_view_iterator;
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

UNIT_TEST_INSTANTIATION( AdjointTally, Normalize )

//---------------------------------------------------------------------------//
// end tstTpetraAdjointTally.cpp
//---------------------------------------------------------------------------//

