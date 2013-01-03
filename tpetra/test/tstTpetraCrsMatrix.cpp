//---------------------------------------------------------------------------//
/*!
 * \file tstTpetraCrsMatrix.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra::CrsMatrix adapter tests.
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

#include <MCLS_MatrixTraits.hpp>
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
#include <Tpetra_CrsMatrix.hpp>

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
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, Typedefs, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<Scalar, scalar_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<LO, local_ordinal_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<GO, global_ordinal_type>::value)
	== true, true );
}

UNIT_TEST_INSTANTIATION( MatrixTraits, Typedefs )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, Clone, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );

    Teuchos::RCP<MatrixType> B = MT::clone( *A );

    TEST_ASSERT( A->getMap()->isSameAs( *(B->getMap()) ) );
}

UNIT_TEST_INSTANTIATION( MatrixTraits, Clone )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, RowVectorClone, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );

    Teuchos::RCP<VectorType> X = MT::cloneVectorFromMatrixRows( *A );

    TEST_ASSERT( A->getRowMap()->isSameAs( *(X->getMap()) ) );

    Teuchos::ArrayRCP<const Scalar> X_view = VT::view( *X );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator view_iterator;
    for ( view_iterator = X_view.begin();
	  view_iterator != X_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 0.0 );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, RowVectorClone )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, ColVectorClone, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    Teuchos::RCP<VectorType> X = MT::cloneVectorFromMatrixCols( *A );

    TEST_ASSERT( A->getColMap()->isSameAs( *(X->getMap()) ) );

    Teuchos::ArrayRCP<const Scalar> X_view = VT::view( *X );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator view_iterator;
    for ( view_iterator = X_view.begin();
	  view_iterator != X_view.end();
	  ++view_iterator )
    {
	TEST_EQUALITY( *view_iterator, 0.0 );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, ColVectorClone )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, Comm, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );

    Teuchos::RCP<const Teuchos::Comm<int> > copy_comm = MT::getComm( *A );

    TEST_EQUALITY( comm->getRank(), copy_comm->getRank() );
    TEST_EQUALITY( comm->getSize(), copy_comm->getSize() );
}

UNIT_TEST_INSTANTIATION( MatrixTraits, Comm )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, GlobalNumRows, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    TEST_EQUALITY( MT::getGlobalNumRows( *A ), global_num_rows );
}

UNIT_TEST_INSTANTIATION( MatrixTraits, GlobalNumRows )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, LocalNumRows, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    TEST_EQUALITY( MT::getLocalNumRows( *A ), local_num_rows );
}

UNIT_TEST_INSTANTIATION( MatrixTraits, LocalNumRows )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, GlobalMaxEntries, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    TEST_EQUALITY( MT::getGlobalMaxNumRowEntries( *A ), 1 );
}

UNIT_TEST_INSTANTIATION( MatrixTraits, GlobalMaxEntries )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, l2g_row, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    int offset = comm->getRank() * local_num_rows;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_EQUALITY( MT::getGlobalRow( *A, i ), i + offset );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, l2g_row )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, g2l_row, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    int offset = comm->getRank() * local_num_rows;
    for ( int i = offset; i < local_num_rows+offset; ++i )
    {
	TEST_EQUALITY( MT::getLocalRow( *A, i ), i );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, g2l_row )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, l2g_col, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    int offset = comm->getRank() * local_num_rows;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_EQUALITY( MT::getGlobalCol( *A, i ), i + offset );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, l2g_col )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, g2l_col, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    int offset = comm->getRank() * local_num_rows;
    for ( int i = offset; i < local_num_rows+offset; ++i )
    {
	TEST_EQUALITY( MT::getLocalCol( *A, i ), i );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, g2l_col )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, is_l_row, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_ASSERT( MT::isLocalRow( *A, i ) );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, is_l_row )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, is_g_row, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    int offset = comm->getRank() * local_num_rows;
    for ( int i = offset; i < local_num_rows+offset; ++i )
    {
	TEST_ASSERT( MT::isGlobalRow( *A, i ) );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, is_g_row )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, is_l_col, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_ASSERT( MT::isLocalCol( *A, i ) );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, is_l_col )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( MatrixTraits, is_g_col, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<Scalar,LO,GO,VectorType> VT;
    typedef MCLS::MatrixTraits<Scalar,LO,GO,VectorType,MatrixType> MT;
    typedef typename MT::scalar_type scalar_type;
    typedef typename MT::local_ordinal_type local_ordinal_type;
    typedef typename MT::global_ordinal_type global_ordinal_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1, 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    int offset = comm->getRank() * local_num_rows;
    for ( int i = offset; i < local_num_rows+offset; ++i )
    {
	TEST_ASSERT( MT::isGlobalCol( *A, i ) );
    }
}

UNIT_TEST_INSTANTIATION( MatrixTraits, is_g_col )

//---------------------------------------------------------------------------//
// end tstTpetraCrsMatrix.cpp
//---------------------------------------------------------------------------//

