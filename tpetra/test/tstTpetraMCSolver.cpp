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
 * \file tstTpetraMCSolver.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra Monte Carlo set solver tests.
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
#include <random>

#include <MCLS_MCSolver.hpp>
#include <MCLS_UniformAdjointSource.hpp>
#include <MCLS_AdjointTally.hpp>
#include <MCLS_AlmostOptimalDomain.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_MatrixTraits.hpp>
#include <MCLS_TpetraAdapter.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

//---------------------------------------------------------------------------//
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MCSolver, solve )
{
    typedef Tpetra::Vector<double,int,long> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<double,int,long> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef std::mt19937 rng_type;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef MCLS::AlmostOptimalDomain<VectorType,MatrixType,rng_type,TallyType> DomainType;
    typedef MCLS::UniformAdjointSource<DomainType> SourceType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<int,long> > map = 
	Tpetra::createUniformContigMap<int,long>( global_num_rows, comm );

    // Build the linear system. This operator is symmetric with a spectral
    // radius less than 1.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<double,int,long>( map );
    Teuchos::Array<long> global_columns( 3 );
    Teuchos::Array<double> values( 3 );
    global_columns[0] = 0;
    global_columns[1] = 1;
    global_columns[2] = 2;
    values[0] = 1.0/comm_size;
    values[1] = -0.14/comm_size;
    values[2] = 0.0/comm_size;
    A->insertGlobalValues( 0, global_columns(), values() );
    for ( int i = 1; i < global_num_rows-1; ++i )
    {
	global_columns[0] = i-1;
	global_columns[1] = i;
	global_columns[2] = i+1;
	values[0] = -0.14/comm_size;
	values[1] = 1.0/comm_size;
	values[2] = -0.14/comm_size;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    global_columns[0] = global_num_rows-3;
    global_columns[1] = global_num_rows-2;
    global_columns[2] = global_num_rows-1;
    values[0] = 0.0/comm_size;
    values[1] = -0.14/comm_size;
    values[2] = 1.0/comm_size;
    A->insertGlobalValues( global_num_rows-1, global_columns(), values() );
    A->fillComplete();

    // Build the LHS. Put a large positive number here to be sure we are
    // clear the vector before solving.
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );
    VT::putScalar( *x, 100.0 );

    // Build the RHS with negative numbers. this gives us a negative
    // solution. 
    Teuchos::RCP<VectorType> b = MT::cloneVectorFromMatrixRows( *A );
    VT::putScalar( *b, -2.0 );

    // Create the solver.
    Teuchos::RCP<Teuchos::ParameterList> plist = 
	Teuchos::rcp( new Teuchos::ParameterList() );
    plist->set<int>("MC Check Frequency", 10);
    plist->set<bool>("Reproducible MC Mode",true);
    plist->set<std::string>("Transport Type", "Global" );
    MCLS::MCSolver<SourceType> solver( comm, comm->getRank(), plist );

    // Build the adjoint domain.
    plist->set<int>( "Overlap Size", 2 );
    Teuchos::RCP<DomainType> domain = Teuchos::rcp( new DomainType( A, x, *plist ) );

    // Create the adjoint source with a set number of histories.
    int mult = 100;
    plist->set<double>("Sample Ratio",mult);
    Teuchos::RCP<SourceType> source = Teuchos::rcp(
	new SourceType( b, domain, *plist ) );

    // Set the Domain.
    solver.setDomain( domain );

    // Set the Source.
    solver.setSource( source );

    // Do solve the problem.
    solver.solve();

    // Check that we got a negative solution.
    Teuchos::ArrayRCP<const double> x_view = 
        VT::view( *x );
    typename Teuchos::ArrayRCP<const double>::const_iterator x_view_it;
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
	TEST_ASSERT( *x_view_it < Teuchos::ScalarTraits<double>::zero() );
    }

    // Now solve the problem with a positive source.
    VT::putScalar( *b, 2.0 );
    solver.setSource( source );
    solver.solve();
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
	TEST_ASSERT( *x_view_it > Teuchos::ScalarTraits<double>::zero() );
    }

    // Reset the domain and solve again with a positive source.
    solver.setDomain( domain );
    solver.setSource( source );
    solver.solve();
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
	TEST_ASSERT( *x_view_it > Teuchos::ScalarTraits<double>::zero() );
    }

    // Reset both and solve with a negative source.
    VT::putScalar( *b, -2.0 );
    solver.setDomain( domain );
    solver.setSource( source );
    solver.solve();
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
	TEST_ASSERT( *x_view_it < Teuchos::ScalarTraits<double>::zero() );
    }
}

//---------------------------------------------------------------------------//
// end tstTpetraMCSolver.cpp
//---------------------------------------------------------------------------//

