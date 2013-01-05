//---------------------------------------------------------------------------//
/*!
 * \file tstTpetraVector.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra vector tests.
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

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map );
    VT::putScalar( *A, 1.0 );
    Teuchos::RCP<VectorType> B = VT::deepCopy( *A );

    MCLS::VectorExport<VectorType> vector_export( A, B );
    vector_export.doExportAdd();
    
    Teuchos::ArrayRCP<const Scalar> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator view_iterator;
    for ( view_iterator = B_view.begin();
	  view_iterator != B_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 2.0 );
    }
}

UNIT_TEST_INSTANTIATION( VectorExport, Add )

//---------------------------------------------------------------------------//
// end tstTpetraVector.cpp
//---------------------------------------------------------------------------//

