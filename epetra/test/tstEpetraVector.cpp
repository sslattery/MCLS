//---------------------------------------------------------------------------//
/*!
 * \file tstEpetraVector.cpp
 * \author Stuart R. Slattery
 * \brief Epetra vector tests.
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

#include <MCLS_VectorTraits.hpp>
#include <MCLS_EpetraAdapter.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_OpaqueWrapper.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>

#include <Epetra_Map.h>
#include <Epetra_Vector.h>
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
// Unit tests.
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorTraits, Typedefs )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef VT::scalar_type scalar_type;
    typedef VT::local_ordinal_type local_ordinal_type;
    typedef VT::global_ordinal_type global_ordinal_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<double, scalar_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<int, local_ordinal_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<int, global_ordinal_type>::value)
	== true, true );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorTraits, Clone )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef VT::scalar_type scalar_type;
    typedef VT::local_ordinal_type local_ordinal_type;
    typedef VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();
    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;

    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map ) );

    VT::putScalar( *A, 1.0 );

    Teuchos::RCP<VectorType> B = VT::clone( *A );

    TEST_ASSERT( A->Map().SameAs( B->Map() ) );
    
    Teuchos::ArrayRCP<const double> B_view = VT::view( *B );
    Teuchos::ArrayRCP<const double>::const_iterator view_iterator;
    for ( view_iterator = B_view.begin();
	  view_iterator != B_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 0.0 );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorTraits, DeepCopy )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef VT::scalar_type scalar_type;
    typedef VT::local_ordinal_type local_ordinal_type;
    typedef VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map ) );
    VT::putScalar( *A, 1.0 );

    Teuchos::RCP<VectorType> B = VT::deepCopy( *A );

    TEST_ASSERT( A->Map().SameAs( B->Map() ) );
    
    Teuchos::ArrayRCP<const double> B_view = VT::view( *B );
    Teuchos::ArrayRCP<const double>::const_iterator view_iterator;
    for ( view_iterator = B_view.begin();
	  view_iterator != B_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 1.0 );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorTraits, Modifiers )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef VT::scalar_type scalar_type;
    typedef VT::local_ordinal_type local_ordinal_type;
    typedef VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map ) );

    VT::putScalar( *A, 2.0 );    

    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    Teuchos::ArrayRCP<const double>::const_iterator view_iterator;
    for ( view_iterator = A_view.begin();
	  view_iterator != A_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 2.0 );
    }

    Teuchos::ArrayRCP<double> A_view_non_const = VT::viewNonConst( *A );
    Teuchos::ArrayRCP<double>::iterator view_non_const_iterator;
    for ( view_non_const_iterator = A_view_non_const.begin();
	  view_non_const_iterator != A_view_non_const.end();
	  ++view_non_const_iterator )
    {
	*view_non_const_iterator = 3.0;
    }

    for ( view_iterator = A_view.begin();
	  view_iterator != A_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 3.0 );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorTraits, SumIntoElement )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef VT::scalar_type scalar_type;
    typedef VT::local_ordinal_type local_ordinal_type;
    typedef VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map ) );
    VT::putScalar( *A, 1.0 );

    Teuchos::ArrayView<const int> global_elements( 
	A->Map().MyGlobalElements(), local_num_rows );
    Teuchos::ArrayView<const int>::const_iterator element_iterator;
    for ( element_iterator = global_elements.begin();
	  element_iterator != global_elements.end();
	  ++element_iterator )
    {
	int local_element = VT::getLocalRow( *A, *element_iterator );
	int global_row = VT::getGlobalRow( *A, local_element );
	TEST_ASSERT( VT::isGlobalRow( *A, global_row ) );
	VT::sumIntoGlobalValue( *A, global_row, 2.0 );
    }

    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    Teuchos::ArrayRCP<const double>::const_iterator view_iterator;
    for ( view_iterator = A_view.begin();
	  view_iterator != A_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 3.0 );
    }

    for ( element_iterator = global_elements.begin();
	  element_iterator != global_elements.end();
	  ++element_iterator )
    {
	int local_element = VT::getLocalRow( *A, *element_iterator );
	TEST_ASSERT( VT::isLocalRow( *A, local_element ) );
	VT::sumIntoLocalValue( *A, local_element, 2.0 );
    }

    for ( view_iterator = A_view.begin();
	  view_iterator != A_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 5.0 );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorTraits, ReplaceElement )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef VT::scalar_type scalar_type;
    typedef VT::local_ordinal_type local_ordinal_type;
    typedef VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map ) );
    VT::putScalar( *A, 1.0 );

    Teuchos::ArrayView<const int> global_elements( 
	A->Map().MyGlobalElements(), local_num_rows );

    Teuchos::ArrayView<const int>::const_iterator element_iterator;
    for ( element_iterator = global_elements.begin();
	  element_iterator != global_elements.end();
	  ++element_iterator )
    {
	int local_element = VT::getLocalRow( *A, *element_iterator );
	int global_row = VT::getGlobalRow( *A, local_element );
	TEST_ASSERT( VT::isGlobalRow( *A, global_row ) );
	VT::replaceGlobalValue( *A, global_row, 2.0 );
    }

    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    Teuchos::ArrayRCP<const double>::const_iterator view_iterator;
    for ( view_iterator = A_view.begin();
	  view_iterator != A_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 2.0 );
    }

    for ( element_iterator = global_elements.begin();
	  element_iterator != global_elements.end();
	  ++element_iterator )
    {
	int local_element = VT::getLocalRow( *A, *element_iterator );
	TEST_ASSERT( VT::isLocalRow( *A, local_element ) );
	VT::replaceLocalValue( *A, local_element, 5.0 );
    }

    for ( view_iterator = A_view.begin();
	  view_iterator != A_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 5.0 );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorTraits, DotProduct )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef VT::scalar_type scalar_type;
    typedef VT::local_ordinal_type local_ordinal_type;
    typedef VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map ) );
    VT::putScalar( *A, 2.0 );

    Teuchos::RCP<VectorType> B = VT::deepCopy( *A );

    double product = 2.0*2.0*global_num_rows;
    TEST_EQUALITY( VT::dot( *A, *B ), product );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorTraits, Norms )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef VT::scalar_type scalar_type;
    typedef VT::local_ordinal_type local_ordinal_type;
    typedef VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map ) );
    VT::putScalar( *A, 2.0 );

    double norm_two = std::pow( 4.0*global_num_rows, 0.5 );
    TEST_EQUALITY( VT::norm2( *A ), norm_two );
    TEST_EQUALITY( VT::norm1( *A ), 2.0*global_num_rows );
    TEST_EQUALITY( VT::normInf( *A ), 2.0 );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorTraits, MeanValue )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef VT::scalar_type scalar_type;
    typedef VT::local_ordinal_type local_ordinal_type;
    typedef VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map ) );
    VT::putScalar( *A, 2.0 );
    VT::replaceLocalValue( *A, 0, 1.0 );

    double mean_value = ((global_num_rows-comm_size)*2.0 + comm_size*1.0) 
			/ global_num_rows;
    TEST_FLOATING_EQUALITY( VT::meanValue( *A ), mean_value, 1.0e-8 );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorTraits, AbsoluteVal )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef VT::scalar_type scalar_type;
    typedef VT::local_ordinal_type local_ordinal_type;
    typedef VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map ) );
    VT::putScalar( *A, -2.0 );

    Teuchos::RCP<VectorType> B = VT::clone( *A );

    VT::abs( *B, *A );
    Teuchos::ArrayRCP<const double> B_view = VT::view( *B );
    Teuchos::ArrayRCP<const double>::const_iterator B_view_iterator;
    for ( B_view_iterator = B_view.begin();
	  B_view_iterator != B_view.end();
	  ++B_view_iterator )
    {
	TEST_EQUALITY( *B_view_iterator, 2.0 );
    }

    VT::abs( *A, *A );
    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    Teuchos::ArrayRCP<const double>::const_iterator A_view_iterator;
    for ( A_view_iterator = A_view.begin();
	  A_view_iterator != A_view.end();
	  ++A_view_iterator )
    {
	TEST_EQUALITY( *A_view_iterator, 2.0 );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorTraits, Scale )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef VT::scalar_type scalar_type;
    typedef VT::local_ordinal_type local_ordinal_type;
    typedef VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map ) );
    VT::putScalar( *A, 2.0 );

    VT::scale( *A, 3.0 );
    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    Teuchos::ArrayRCP<const double>::const_iterator A_view_iterator;
    for ( A_view_iterator = A_view.begin();
	  A_view_iterator != A_view.end();
	  ++A_view_iterator )
    {
	TEST_EQUALITY( *A_view_iterator, 6.0 );
    }

    Teuchos::RCP<VectorType> B = VT::clone( *A );
    VT::scaleCopy( *B, 2.0, *A );
    Teuchos::ArrayRCP<const double> B_view = VT::view( *B );
    Teuchos::ArrayRCP<const double>::const_iterator B_view_iterator;
    for ( B_view_iterator = B_view.begin();
	  B_view_iterator != B_view.end();
	  ++B_view_iterator )
    {
	TEST_EQUALITY( *B_view_iterator, 12.0 );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorTraits, Reciprocal )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef VT::scalar_type scalar_type;
    typedef VT::local_ordinal_type local_ordinal_type;
    typedef VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map ) );
    VT::putScalar( *A, 2.0 );

    Teuchos::RCP<VectorType> B = VT::clone( *A );

    double recip_val = 1 / 2.0;
    VT::reciprocal( *B, *A );
    Teuchos::ArrayRCP<const double> B_view = VT::view( *B );
    Teuchos::ArrayRCP<const double>::const_iterator B_view_iterator;
    for ( B_view_iterator = B_view.begin();
	  B_view_iterator != B_view.end();
	  ++B_view_iterator )
    {
	TEST_EQUALITY( *B_view_iterator, recip_val );
    }

    VT::reciprocal( *A, *A );
    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    Teuchos::ArrayRCP<const double>::const_iterator A_view_iterator;
    for ( A_view_iterator = A_view.begin();
	  A_view_iterator != A_view.end();
	  ++A_view_iterator )
    {
	TEST_EQUALITY( *A_view_iterator, recip_val );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorTraits, Update )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef VT::scalar_type scalar_type;
    typedef VT::local_ordinal_type local_ordinal_type;
    typedef VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map ) );
    VT::putScalar( *A, 1.0 );

    Teuchos::RCP<VectorType> B = VT::clone( *A );
    VT::putScalar( *B, 2.0 );

    double alpha = 4.0;
    double beta = 5.0;

    double update_val = alpha*1.0 + beta*2.0;
    VT::update( *A, alpha, *B, beta );
    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    Teuchos::ArrayRCP<const double>::const_iterator A_view_iterator;
    for ( A_view_iterator = A_view.begin();
	  A_view_iterator != A_view.end();
	  ++A_view_iterator )
    {
	TEST_EQUALITY( *A_view_iterator, update_val );
    }

    Teuchos::RCP<VectorType> C = VT::clone( *A );
    VT::putScalar( *C, 3.0 );
    double gamma = 6.0;
    update_val = update_val*alpha + beta*2.0 + gamma*3.0;
    VT::update( *A, alpha, *B, beta, *C, gamma );
    for ( A_view_iterator = A_view.begin();
	  A_view_iterator != A_view.end();
	  ++A_view_iterator )
    {
	TEST_EQUALITY( *A_view_iterator, update_val );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorTraits, ElementWiseMultiply )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef VT::scalar_type scalar_type;
    typedef VT::local_ordinal_type local_ordinal_type;
    typedef VT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map ) );
    VT::putScalar( *A, 1.0 );

    Teuchos::RCP<VectorType> B = VT::clone( *A );
    VT::putScalar( *B, 2.0 );

    Teuchos::RCP<VectorType> C = VT::clone( *A );
    VT::putScalar( *C, 3.0 );

    double alpha = 4.0;
    double beta = 5.0;

    double multiply_val = 1.0*alpha + beta*2.0*3.0;
    VT::elementWiseMultiply( *A, alpha, *B, *C, beta );

    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    Teuchos::ArrayRCP<const double>::const_iterator A_view_iterator;
    for ( A_view_iterator = A_view.begin();
	  A_view_iterator != A_view.end();
	  ++A_view_iterator )
    {
	TEST_EQUALITY( *A_view_iterator, multiply_val );
    }
}

//---------------------------------------------------------------------------//
// end tstEpetraVector.cpp
//---------------------------------------------------------------------------//

