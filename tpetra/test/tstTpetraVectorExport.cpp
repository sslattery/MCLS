//---------------------------------------------------------------------------//
/*!
 * \file tstTpetraVectorExport.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra VectorExport tests.
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

#include <MCLS_VectorExport.hpp>
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
#include <Tpetra_Export.hpp>

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
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorExport, Typedefs, LO, GO, Scalar )
{
    typedef Tpetra::Export<LO,GO> ExportType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef MCLS::VectorExport<VectorType> VE;
    typedef typename VE::export_type export_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<ExportType, export_type>::value)
	== true, true );
}

UNIT_TEST_INSTANTIATION( VectorExport, Typedefs )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorExport, Add, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef typename VT::scalar_type scalar_type;
    typedef typename VT::local_ordinal_type local_ordinal_type;
    typedef typename VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_a = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map_a );
    Scalar a_val = 2;
    VT::putScalar( *A, a_val );

    Teuchos::Array<GO> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_b = 
	Tpetra::createNonContigMap<LO,GO>( inverse_rows(), comm );

    Teuchos::RCP<VectorType> B = Tpetra::createVector<Scalar,LO,GO>( map_b );
    Scalar b_val = 3;
    VT::putScalar( *B, b_val );

    MCLS::VectorExport<VectorType> vector_export( A, B );
    vector_export.doExportAdd();
    
    Teuchos::ArrayRCP<const Scalar> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator view_iterator;
    for ( view_iterator = B_view.begin();
	  view_iterator != B_view.end();
	  ++view_iterator )
    {
	if ( comm_size == 1 )
	{
	    TEST_EQUALITY( *view_iterator, a_val );
	}
	else
	{
	    TEST_EQUALITY( *view_iterator, b_val + a_val );
	}
    }
}

UNIT_TEST_INSTANTIATION( VectorExport, Add )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( VectorExport, Insert, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef typename VT::scalar_type scalar_type;
    typedef typename VT::local_ordinal_type local_ordinal_type;
    typedef typename VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_a = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map_a );
    Scalar a_val = 2;
    VT::putScalar( *A, a_val );

    Teuchos::Array<GO> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_b = 
	Tpetra::createNonContigMap<LO,GO>( inverse_rows(), comm );

    Teuchos::RCP<VectorType> B = Tpetra::createVector<Scalar,LO,GO>( map_b );
    Scalar b_val = 3;
    VT::putScalar( *B, b_val );

    MCLS::VectorExport<VectorType> vector_export( A, B );
    vector_export.doExportInsert();
    
    Teuchos::ArrayRCP<const Scalar> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator view_iterator;
    for ( view_iterator = B_view.begin();
	  view_iterator != B_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, a_val );
    }
}

UNIT_TEST_INSTANTIATION( VectorExport, Insert )

//---------------------------------------------------------------------------//
// end tstTpetraVectorExport.cpp
//---------------------------------------------------------------------------//

