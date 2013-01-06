//---------------------------------------------------------------------------//
/*!
 * \file tstEpetraVectorExport.cpp
 * \author Stuart R. Slattery
 * \brief Epetra VectorExport tests.
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
#include <Epetra_Export.h>

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
TEUCHOS_UNIT_TEST( VectorExport, Typedefs )
{
    typedef Epetra_Export ExportType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<double,int,int,VectorType> VT;
    typedef MCLS::VectorExport<VectorType> VE;
    typedef typename VE::export_type export_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<ExportType, export_type>::value)
	== true, true );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorExport, Add )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<double,int,int,VectorType> VT;
    typedef typename VT::scalar_type scalar_type;
    typedef typename VT::local_ordinal_type local_ordinal_type;
    typedef typename VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map ) );
    double a_val = 2;
    VT::putScalar( *A, a_val );

    Teuchos::RCP<VectorType> B = VT::clone( *A );
    double b_val = 3;
    VT::putScalar( *B, b_val );

    MCLS::VectorExport<VectorType> vector_export( A, B );
    vector_export.doExportAdd();
    
    Teuchos::ArrayRCP<const double> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const double>::const_iterator view_iterator;
    for ( view_iterator = B_view.begin();
	  view_iterator != B_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, b_val + a_val );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorExport, Insert )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<double,int,int,VectorType> VT;
    typedef typename VT::scalar_type scalar_type;
    typedef typename VT::local_ordinal_type local_ordinal_type;
    typedef typename VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map ) );
    double a_val = 2;
    VT::putScalar( *A, a_val );

    Teuchos::RCP<VectorType> B = VT::clone( *A );
    double b_val = 3;
    VT::putScalar( *B, b_val );

    MCLS::VectorExport<VectorType> vector_export( A, B );
    vector_export.doExportInsert();
    
    Teuchos::ArrayRCP<const double> B_view = VT::view( *B );
    typename Teuchos::ArrayRCP<const double>::const_iterator view_iterator;
    for ( view_iterator = B_view.begin();
	  view_iterator != B_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, b_val );
    }
}

//---------------------------------------------------------------------------//
// end tstEpetraVectorExport.cpp
//---------------------------------------------------------------------------//

