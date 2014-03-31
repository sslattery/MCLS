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

    // Problem parameters.
    int problem_size = plist->get<int>("Problem Size");

    // Create the poisson operator.
    Teuchos::RCP<Epetra_CrsMatrix> A =
	buildPoissonOperator( problem_size, epetra_comm );
    
    // Create the solution vector.
    Teuchos::RCP<Epetra_Vector> u = MT::cloneVectorFromMatrixRows( *A );

    // Create the initial guess.
    int k_1 = plist->get<int>("Wave Number 1");
    int k_2 = plist->get<int>("Wave Number 2");
    VT::putScalar( *u, 0.0 );
    double pi = std::acos(-1.0);
    for ( int i = 1; i < problem_size-1; ++i )
    {
	(*u)[i] = std::sin( i*k_1*pi / (problem_size-1) ) +
		  std::sin( i*k_2*pi / (problem_size-1) );
    }

    // Build the correction vector .
    Teuchos::RCP<Epetra_Vector> d = MT::cloneVectorFromMatrixRows( *A );
    VT::putScalar( *d, 0.0 );

    // Build the residual.
    Teuchos::RCP<Epetra_Vector> r = MT::cloneVectorFromMatrixRows( *A );
    MT::apply( *A, *u, *r );
    VT::scale( *r, -1.0 );

    // Create the residual linear problem.
    Teuchos::RCP<MCLS::LinearProblem<Vector,Matrix> > linear_problem =
	Teuchos::rcp( new MCLS::LinearProblem<Vector,Matrix>(
			  A, d, r ) );

    // Create the solver.
    std::string solver_type = plist->get<std::string>("Solver Type");
    MCLS::SolverFactory<Vector,Matrix> factory;
    Teuchos::RCP<MCLS::SolverManager<Vector,Matrix> > solver_manager =
	factory.create( solver_type, comm, plist );

    // Solve the problem.
    std::cout << std::endl;
    Teuchos::Time timer("");
    timer.start(true);
    solver_manager->setParameters( plist );
    solver_manager->setProblem( linear_problem );
    solver_manager->solve();
    timer.stop();
    std::cout << std::endl;

    // Apply the correction to the solution.
    VT::update( *u, 1.0, *d, 1.0 );

    // The point-wise error is the inf-norm of the top level solution.
    double e_inf = VT::normInf( *u );
    std::cout << "||e||_inf: " << e_inf << std::endl;

    // Compute the figure of merit -> error * work
    double toc = timer.totalElapsedTime();
    std::cout << "Time: " << toc << std::endl;
    double fom = 1.0 / (e_inf * e_inf * toc);
    std::cout << "Figure of merit: " << fom << std::endl;
    std::cout << std::endl;

    // Write the solution to a file.
    if ( comm->getRank() == 0 )
    {
        std::ofstream ofile;
        ofile.open( "solution.dat" );
        for ( int i = 0; i < problem_size; ++i )
        {
            ofile << std::setprecision(8) << (*d)[i] << std::endl;
        }
        ofile.close();
    }
    comm->barrier();

    return 0;
}

//---------------------------------------------------------------------------//
// end neutron_diffusion.cpp
//---------------------------------------------------------------------------//

