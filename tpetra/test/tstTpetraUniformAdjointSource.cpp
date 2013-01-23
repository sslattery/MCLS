//---------------------------------------------------------------------------//
/*!
 * \file tstTpetraUniformAdjointSource.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra AdjointDomain tests.
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

#include <MCLS_UniformAdjointSource.hpp>
#include <MCLS_AdjointDomain.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_TpetraAdapter.hpp>
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

#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

//---------------------------------------------------------------------------//
// Instantiation macro. 
// 
// These types are those enabled by Tpetra under explicit instantiation. I
// have removed scalar types that are not floating point
//---------------------------------------------------------------------------//
#define UNIT_TEST_INSTANTIATION( type, name )			           \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( type, name, int, int, double )   \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( type, name, int, long, double )

//---------------------------------------------------------------------------//
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( UniformAdjointSource, Typedefs, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;
    typedef MCLS::History<GO> HistoryType;

    typedef MCLS::UniformAdjointSource<DomainType> SourceType;
    typedef typename SourceType::HistoryType history_type;
    typedef typename SourceType::VectorType vector_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<HistoryType, history_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<VectorType, vector_type>::value)
	== true, true );
}

UNIT_TEST_INSTANTIATION( UniformAdjointSource, Typedefs )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( UniformAdjointSource, nh_not_set, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::History<GO> HistoryType;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    // Build the linear system.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1 );
    for ( int i = 1; i < global_num_rows; ++i )
    {
	global_columns[0] = i-1;
	values[0] = -0.5/comm_size;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    global_columns[0] = global_num_rows-1;
    values[0] = -0.5/comm_size;
    A->insertGlobalValues( global_num_rows-1, global_columns(), values() );
    A->fillComplete();

    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );
    Teuchos::RCP<VectorType> b = MT::cloneVectorFromMatrixRows( *A );
    VT::putScalar( *b, -1.0 );

    // Build the adjoint domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 0 );
    Teuchos::RCP<DomainType> domain = Teuchos::rcp( new DomainType( A, x, plist ) );

    // History setup.
    Teuchos::RCP<MCLS::RNGControl> control = Teuchos::rcp(
	new MCLS::RNGControl( 3939294 ) );
    HistoryType::setByteSize( control->getSize() );

    // Create the adjoint source with default values.
    double cutoff = 1.0e-8;
    plist.set<double>("Weight Cutoff", cutoff );
    MCLS::UniformAdjointSource<DomainType> 
	source( b, domain, control, comm, plist );
    TEST_ASSERT( source.empty() );
    TEST_EQUALITY( source.numToTransport(), 0 );
    TEST_EQUALITY( source.numToTransportInSet(), global_num_rows );
    TEST_EQUALITY( source.numRequested(), global_num_rows );
    TEST_EQUALITY( source.numLeft(), 0 );
    TEST_EQUALITY( source.numEmitted(), 0 );
    TEST_EQUALITY( source.numStreams(), 0 );
    TEST_ASSERT( plist.isParameter("Relative Weight Cutoff") );
    TEST_EQUALITY( plist.get<double>("Relative Weight Cutoff"),
		   global_num_rows*cutoff );

    // Build the source.
    source.buildSource();
    TEST_ASSERT( !source.empty() );
    TEST_EQUALITY( source.numToTransport(), local_num_rows );
    TEST_EQUALITY( source.numToTransportInSet(), global_num_rows );
    TEST_EQUALITY( source.numRequested(), global_num_rows );
    TEST_EQUALITY( source.numLeft(), local_num_rows );
    TEST_EQUALITY( source.numEmitted(), 0 );
    TEST_EQUALITY( source.numStreams(), comm->getSize() );

    // Sample the source.
    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_ASSERT( !source.empty() );
	TEST_EQUALITY( source.numLeft(), local_num_rows-i );
	TEST_EQUALITY( source.numEmitted(), i );

	Teuchos::RCP<HistoryType> history = source.getHistory();

	TEST_EQUALITY( history->weight(), -global_num_rows );
	TEST_ASSERT( domain->isLocalState( history->state() ) );
	TEST_ASSERT( history->alive() );
	TEST_ASSERT( VT::isGlobalRow( *x, history->state() ) );
    }
    TEST_ASSERT( source.empty() );
    TEST_EQUALITY( source.numLeft(), 0 );
    TEST_EQUALITY( source.numEmitted(), local_num_rows );
}

UNIT_TEST_INSTANTIATION( UniformAdjointSource, nh_not_set )

//---------------------------------------------------------------------------//
// end tstTpetraUniformAdjointSource.cpp
//---------------------------------------------------------------------------//
