//---------------------------------------------------------------------------//
/*!
 * \file poisson_driver.cpp
 * \author Stuart R. Slattery
 * \brief 1D poisson driver.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <string>
#include <cassert>

#include <MCLS_SolverFactory.hpp>
#include <MCLS_SolverManager.hpp>
#include <MCLS_LinearProblem.hpp>
#include <MCLS_EpetraAdapter.hpp>

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_as.hpp>
#include <Teuchos_OpaqueWrapper.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_Tuple.hpp>
#include <Teuchos_ParameterList.hpp>
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_XMLParameterListCoreHelpers.hpp"
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_Time.hpp>

#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_Comm.h>
#include <Epetra_SerialComm.h>

#ifdef HAVE_MPI
#include <Epetra_MpiComm.h>
#include <Teuchos_DefaultMpiComm.hpp>
#endif

//---------------------------------------------------------------------------//
// Typedefs.
//---------------------------------------------------------------------------//

typedef Epetra_Vector Vector;
typedef MCLS::VectorTraits<Vector> VT;
typedef Epetra_RowMatrix Matrix;
typedef MCLS::MatrixTraits<Vector,Matrix> MT;

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
// Poisson operator.
Teuchos::RCP<Epetra_CrsMatrix> buildPoissonOperator(
    const int local_num_rows,
    const Teuchos::RCP<Epetra_Comm> epetra_comm )
{
    // Setup parallel distribution.
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( local_num_rows, 0, *epetra_comm ) );

    // Build the linear operator.
    Teuchos::RCP<Epetra_CrsMatrix> A = 	
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
    Teuchos::Array<int> global_columns( 3 );
    Teuchos::Array<double> values( 3 );
    global_columns[0] = 0;
    values[0] = 1.0;
    A->InsertGlobalValues( 0, 1, &values[0], &global_columns[0] );
    for ( int i = 1; i < local_num_rows-1; ++i )
    {
	global_columns[0] = i-1;
	global_columns[1] = i;
	global_columns[2] = i+1;
	values[0] = -0.5;
	values[1] = 1.0;
	values[2] = -0.5;
	A->InsertGlobalValues( i, 3, 
			       &values[0], &global_columns[0] );
    }
    global_columns[0] = local_num_rows-1;
    values[0] = 1.0;
    A->InsertGlobalValues( local_num_rows-1, 1, 
			   &values[0], &global_columns[0] );
    A->FillComplete();
    return A;
}

//---------------------------------------------------------------------------//
// Prolongation operator. 2h -> h. 
void applyProlongationOperator2( const Teuchos::ArrayView<const double>& v_2h,
				Teuchos::ArrayView<double>& v_h )
{
    assert( v_h.size() == 2 * v_2h.size() - 1 );

    for ( unsigned j = 0; j < v_2h.size() - 1; ++j )
    {
	v_h[2*j] = v_2h[j];
	v_h[2*j+1] = 0.5 * ( v_2h[j] + v_2h[j+1] );
    }

    v_h.back() = v_2h.back();
}

//---------------------------------------------------------------------------//
// Restriction operator. h -> 2h.
void applyRestrictionOperator2( const Teuchos::ArrayView<const double>& v_h,
			       Teuchos::ArrayView<double>& v_2h )
{
    assert( v_h.size() == 2 * v_2h.size() - 1 );

    v_2h.front() = v_h.front();

    for ( unsigned j = 1; j < v_2h.size() - 1; ++j )
    {
	v_2h[j] = 0.25 * ( v_h[2*j-1] + 2 * v_h[2*j] + v_h[2*j+1] );
    }

    v_2h.back() = v_h.back();
}

//---------------------------------------------------------------------------//
// Prolongation operator. 4h -> h. 
void applyProlongationOperator4( const Teuchos::ArrayView<const double>& v_4h,
				Teuchos::ArrayView<double>& v_h )
{
    assert( v_h.size() == 4*(v_4h.size()-1) + 1 );

    for ( unsigned j = 0; j < v_4h.size() - 1; ++j )
    {
	v_h[4*j] = v_4h[j];
	v_h[4*j+1] = 0.75*v_4h[j] + 0.25*v_4h[j+1];
	v_h[4*j+2] = 0.5*v_4h[j] + 0.5*v_4h[j+1];
	v_h[4*j+3] = 0.25*v_4h[j] + 0.75*v_4h[j+1];
    }

    v_h.back() = v_4h.back();
}

//---------------------------------------------------------------------------//
// Restriction operator. h -> 4h.
void applyRestrictionOperator4( const Teuchos::ArrayView<const double>& v_h,
			       Teuchos::ArrayView<double>& v_4h )
{
    assert( v_h.size() == 4*(v_4h.size()-1) + 1 );

    v_4h.front() = v_h.front();

    for ( unsigned j = 1; j < v_4h.size() - 1; ++j )
    {
	v_4h[j] = 0.1 * ( v_h[4*j-2] + 2*v_h[4*j-1] + 4*v_h[4*j] +
			    2*v_h[4*j+1] + v_h[4*j+2] );
    }

    v_4h.back() = v_h.back();
}

//---------------------------------------------------------------------------//
// Prolongation operator.
void applyProlongationOperator( const int M,
				const Teuchos::ArrayView<const double>& v_Mh,
				Teuchos::ArrayView<double>& v_h )
{
    if ( 2 == M )
    {
	applyProlongationOperator2( v_Mh, v_h );
    }

    else if ( 4 == M )
    {
	applyProlongationOperator4( v_Mh, v_h );
    }
}

//---------------------------------------------------------------------------//
// Restriction operator.
void applyRestrictionOperator( const int M,
			       const Teuchos::ArrayView<const double>& v_h,
			       Teuchos::ArrayView<double>& v_Mh )
{
    if ( 2 == M )
    {
	applyRestrictionOperator2( v_h, v_Mh );
    }

    else if ( 4 == M )
    {
	applyRestrictionOperator4( v_h, v_Mh );
    }
}

//---------------------------------------------------------------------------//
int main( int argc, char * argv[] )
{
    // Initialize parallel communication.
    Teuchos::GlobalMPISession mpi_session( &argc, &argv );
    Teuchos::RCP<const Teuchos::Comm<int> > comm =
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    // Initialize output stream.
    Teuchos::FancyOStream out( Teuchos::rcpFromRef( std::cout ) );
    out.setOutputToRootOnly( 0 );
    out.setShowProcRank( true );

    // Read in command line options.
    std::string xml_input_filename;
    Teuchos::CommandLineProcessor clp(false);
    clp.setOption( "xml-in-file",
		   &xml_input_filename,
		   "The XML file to read into a parameter list" );
    clp.parse(argc,argv);

    // Build the parameter list from the xml input.
    Teuchos::RCP<Teuchos::ParameterList> plist =
	Teuchos::rcp( new Teuchos::ParameterList() );
    Teuchos::updateParametersFromXmlFile(
	xml_input_filename, Teuchos::inoutArg(*plist) );

    // Hierarchy parameters.
    int problem_size = plist->get<int>("Problem Size");
    int num_levels = plist->get<int>("Number of Levels");
    int set_histories = plist->get<int>("Set Number of Histories");
    int M = plist->get<int>("Grid Refinement");

    // Create the number of histories hierarchy.
    Teuchos::Array<int> num_histories( num_levels );
    for ( int n = 0; n < num_levels; ++n )
    {
	num_histories[n] =
	    set_histories * std::pow( M, -3.0*(num_levels-n-1)/2.0 );
    }

    // Create the grid hierarchy.
    Teuchos::Array<int> grid_sizes( num_levels, problem_size );
    for ( int n = 1; n < num_levels; ++n )
    {
	grid_sizes[n] = (grid_sizes[n-1] - 1) / M + 1;
    }

    // Create the poisson operator hierarchy.
    Teuchos::Array<Teuchos::RCP<Epetra_CrsMatrix> > A(num_levels);
    for ( int n = 0; n < num_levels; ++n )
    {
	A[n] = buildPoissonOperator( grid_sizes[n], epetra_comm );
    }
    
    // Create a vector hierarchy.
    int k_1 = plist->get<int>("Wave Number 1");
    int k_2 = plist->get<int>("Wave Number 2");
    double pi = std::acos(-1.0);
    Teuchos::Array<Teuchos::RCP<Epetra_Vector> > u(num_levels);
    for ( int n = 0; n < num_levels; ++n )
    {
	u[n] = MT::cloneVectorFromMatrixRows( *A[n] );
	VT::putScalar( *u[n], 0.0 );
	for ( int i = 1; i < grid_sizes[n]-1; ++i )
	{
	    (*u[n])[i] = std::sin( i*k_1*pi / (grid_sizes[n]-1) ) +
			 std::sin( i*k_2*pi / (grid_sizes[n]-1) );
	}
    }

    // Check that R_0*A_0*P_0*x = A_1*x
    for ( int n = 0; n < num_levels - 1; ++n )
    {
	// Do R*A*P*x
	Teuchos::Array<double> RAPx( grid_sizes[n+1], 0.0 );
	{
	    // Get x.
	    double* u_h_ptr;
	    u[n+1]->ExtractView( &u_h_ptr );
	    Teuchos::ArrayView<const double> u_h( u_h_ptr, grid_sizes[n+1] );

	    // Apply P.
	    Teuchos::RCP<Vector> work_1 = VT::clone( *u[n] );
	    double* work_1_ptr;
	    work_1->ExtractView( &work_1_ptr );
	    Teuchos::ArrayView<double> work_1_view( work_1_ptr, grid_sizes[n] );
	    applyProlongationOperator( M, u_h, work_1_view );

	    // Apply A.
	    Teuchos::RCP<Vector> work_2 = VT::clone( *u[n] );
	    MT::apply( *A[n], *work_1, *work_2 );

	    // Apply R.
	    double* work_2_ptr;
	    work_2->ExtractView( &work_2_ptr );
	    Teuchos::ArrayView<const double> work_2_view( 
		work_2_ptr, grid_sizes[n] );
	    Teuchos::ArrayView<double> RAPx_view = RAPx();
	    applyRestrictionOperator( M, work_2_view, RAPx_view );
	}

	// Do A*x
	Teuchos::RCP<Vector> Ax = VT::clone( *u[n+1] );
	MT::apply( *A[n+1], *u[n+1], *Ax );

	// Check the result.
	std::cout << "R*A*P*x = A*x " << n << std::endl;
	for ( int i = 0; i < grid_sizes[n+1]; ++i )
	{
	    std::cout << RAPx[i] << " " << (*Ax)[i] << " " 
		      << (*Ax)[i] / RAPx[i] << std::endl;
	}
	std::cout << std::endl;
    }

    // Check that R_0*P_0*x = x
    for ( int n = 0; n < num_levels - 1; ++n )
    {
	// Do R*P*x
	Teuchos::Array<double> RPx( grid_sizes[n+1], 0.0 );
	{
	    // Get x.
	    double* u_h_ptr;
	    u[n+1]->ExtractView( &u_h_ptr );
	    Teuchos::ArrayView<const double> u_h( u_h_ptr, grid_sizes[n+1] );

	    // Apply P.
	    Teuchos::RCP<Vector> work_1 = VT::clone( *u[n] );
	    double* work_1_ptr;
	    work_1->ExtractView( &work_1_ptr );
	    Teuchos::ArrayView<double> work_1_view( work_1_ptr, grid_sizes[n] );
	    applyProlongationOperator( M, u_h, work_1_view );

	    // Apply R.
	    Teuchos::ArrayView<double> RPx_view = RPx();
	    applyRestrictionOperator( M, work_1_view, RPx_view );
	}

	// Check the result.
	std::cout << "R*P*x, x, c " << n << std::endl;
	for ( int i = 0; i < grid_sizes[n+1]; ++i )
	{
	    std::cout << RPx[i] << " " << (*u[n+1])[i] << " " 
		      << RPx[i] / (*u[n+1])[i] << std::endl;
	}
	std::cout << std::endl;
    }

    // Determine c for P*R*x = c*x
    for ( int n = 0; n < num_levels - 1; ++n )
    {
	// Do P*R*x
	Teuchos::Array<double> PRx( grid_sizes[n], 0.0 );
	{
	    // Get x.
	    double* u_h_ptr;
	    u[n]->ExtractView( &u_h_ptr );
	    Teuchos::ArrayView<const double> u_h( u_h_ptr, grid_sizes[n] );

	    // Apply R.
	    Teuchos::RCP<Vector> work_1 = VT::clone( *u[n+1] );
	    double* work_1_ptr;
	    work_1->ExtractView( &work_1_ptr );
	    Teuchos::ArrayView<double> work_1_view( work_1_ptr, grid_sizes[n+1] );
	    applyRestrictionOperator( M, u_h, work_1_view );

	    // Apply P.
	    Teuchos::ArrayView<double> PRx_view = PRx();
	    applyProlongationOperator( M, work_1_view, PRx_view );
	}

	// Check the result.
	std::cout << "P*R*x, x, c " << n << std::endl;
	for ( int i = 0; i < grid_sizes[n]; ++i )
	{
	    std::cout << PRx[i] << " " << (*u[n])[i] << " " 
		      << PRx[i] / (*u[n])[i] << std::endl;
	}
	std::cout << std::endl;
    }

    // Check norm preservation of the restriction operators.
    for ( int n = 0; n < num_levels - 1; ++n )
    {
	// Do R*x
	Teuchos::RCP<Vector> Rx = VT::clone( *u[n+1] );
	{
	    // Get x.
	    double* u_h_ptr;
	    u[n]->ExtractView( &u_h_ptr );
	    Teuchos::ArrayView<const double> u_h( u_h_ptr, grid_sizes[n] );

	    // Apply R.
	    double* Rx_ptr;
	    Rx->ExtractView( &Rx_ptr );
	    Teuchos::ArrayView<double> Rx_view( Rx_ptr, grid_sizes[n+1] );
	    applyRestrictionOperator( M, u_h, Rx_view );
	}

	// Check the result.
	std::cout << "||R*x|| = ||x|| " << n << std::endl;
	std::cout << "Inf " << VT::normInf( *Rx ) << " " 
		  << VT::normInf( *u[n] ) << std::endl;
	std::cout << "1 " << VT::norm1( *Rx ) << " " 
		  << VT::norm1( *u[n] ) << std::endl;
	std::cout << "2 " << VT::norm2( *Rx ) << " " 
		  << VT::norm2( *u[n] ) << std::endl;
	std::cout << std::endl;
    }

    // Write the solution to a file.
    if ( comm->getRank() == 0 )
    {
        std::ofstream ofile;
        ofile.open( "solution.dat" );
        for ( int i = 0; i < grid_sizes[0]; ++i )
        {
            ofile << std::setprecision(8) << (*u[0])[i] << std::endl;
        }
        ofile.close();
    }
    comm->barrier();

    return 0;
}

//---------------------------------------------------------------------------//
// end neutron_diffusion.cpp
//---------------------------------------------------------------------------//

