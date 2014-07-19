//---------------------------------------------------------------------------//
/*
  Copyright (c) 2012, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the University of Wisconsin - Madison nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
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
    typedef MCLS::VectorTraits<VectorType> VT;

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
    typedef MCLS::VectorTraits<VectorType> VT;

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

