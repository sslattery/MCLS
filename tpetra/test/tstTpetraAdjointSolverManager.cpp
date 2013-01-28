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
 * \file tstTpetraAdjointSolverManager.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra Adjoint Monte Carlo solver manager tests.
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

#include <MCLS_AdjointSolverManager.hpp>
#include <MCLS_LinearProblem.hpp>
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
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( AdjointSolverManager, one_by_one, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::History<GO> HistoryType;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;
    typedef MCLS::UniformAdjointSource<DomainType> SourceType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    // Build the linear system. This operator is symmetric with a spectral
    // radius less than 1.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 3 );
    Teuchos::Array<Scalar> values( 3 );
    global_columns[0] = 0;
    global_columns[1] = 1;
    global_columns[2] = 2;
    values[0] = 0.24/comm_size;
    values[1] = 0.24/comm_size;
    values[2] = 1.0/comm_size;
    A->insertGlobalValues( 0, global_columns(), values() );
    for ( int i = 1; i < global_num_rows-1; ++i )
    {
	global_columns[0] = i-1;
	global_columns[1] = i;
	global_columns[2] = i+1;
	values[0] = 0.24/comm_size;
	values[1] = 1.0/comm_size;
	values[2] = 0.24/comm_size;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    global_columns[0] = global_num_rows-3;
    global_columns[1] = global_num_rows-2;
    global_columns[2] = global_num_rows-1;
    values[0] = 0.24/comm_size;
    values[1] = 0.24/comm_size;
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
    VT::putScalar( *b, -1.0 );

    // Solver parameters.
    Teuchos::RCP<Teuchos::ParameterList> plist = 
	Teuchos::rcp( new Teuchos::ParameterList() );
    double cutoff = 1.0e-6;
    plist->set<double>("Weight Cutoff", cutoff);
    plist->set<int>("MC Check Frequency", 10);
    plist->set<bool>("Reproducible MC Mode",true);
    plist->set<int>("Overlap Size", 2);
    plist->set<int>("Number of Sets", 1);

    // Create the linear problem.
    Teuchos::RCP<MCLS::LinearProblem<VectorType,MatrixType> > linear_problem =
	Teuchos::rcp( new MCLS::LinearProblem<VectorType,MatrixType>(
			  A, x, b ) );

    // Create the solver.
    MCLS::AdjointSolverManager solver_manager( linear_problem, comm, plist );

    // Solve the problem.
    solver_manager.solve();
    std::cout << "TOL " << solver_manger.acheivedTol() << std::endl;

    // Check that we got a negative solution.
    Teuchos::ArrayRCP<const Scalar> x_view = VT::view(*x);
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator x_view_it;
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
	TEST_ASSERT( *x_view_it < Teuchos::ScalarTraits<Scalar>::zero() );
    }

    // // Now solve the problem with a positive source.
    // VT::putScalar( *b, 2.0 );
    // for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    // {
    // 	TEST_ASSERT( *x_view_it > Teuchos::ScalarTraits<Scalar>::zero() );
    // }

    // // Reset the domain and solve again with a positive source.
    // for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    // {
    // 	TEST_ASSERT( *x_view_it > Teuchos::ScalarTraits<Scalar>::zero() );
    // }

    // // Reset both and solve with a negative source.
    // VT::putScalar( *b, -2.0 );
    // for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    // {
    // 	TEST_ASSERT( *x_view_it < Teuchos::ScalarTraits<Scalar>::zero() );
    // }
}

UNIT_TEST_INSTANTIATION( AdjointSolverManager, one_by_one )

//---------------------------------------------------------------------------//
// end tstTpetraAdjointSolverManager.cpp
//---------------------------------------------------------------------------//

