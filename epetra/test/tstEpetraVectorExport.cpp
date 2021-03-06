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
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorExport, Typedefs )
{
    typedef Epetra_Export ExportType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorExport<VectorType> VE;
    typedef VE::export_type export_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<ExportType, export_type>::value)
	== true, true );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorExport, Add )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

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
    double a_val = 2;
    VT::putScalar( *A, a_val );

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
    double b_val = 3;
    VT::putScalar( *B, b_val );

    MCLS::VectorExport<VectorType> vector_export( A, B );
    vector_export.doExportAdd();

    Teuchos::ArrayRCP<const double> B_view = VT::view( *B );
    Teuchos::ArrayRCP<const double>::const_iterator view_iterator;
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

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( VectorExport, Insert )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;

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
    double a_val = 2;
    VT::putScalar( *A, a_val );

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
    double b_val = 3;
    VT::putScalar( *B, b_val );

    MCLS::VectorExport<VectorType> vector_export( A, B );
    vector_export.doExportInsert();

    Teuchos::ArrayRCP<const double> B_view = VT::view( *B );
    Teuchos::ArrayRCP<const double>::const_iterator view_iterator;
    for ( view_iterator = B_view.begin();
	  view_iterator != B_view.end();
	  ++view_iterator )
    {
	    TEST_EQUALITY( *view_iterator, a_val );
    }
}

//---------------------------------------------------------------------------//
// end tstEpetraVectorExport.cpp
//---------------------------------------------------------------------------//

