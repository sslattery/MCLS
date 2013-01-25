//---------------------------------------------------------------------------//
/*!
 * \file tstEpetraDomainCommunicator.cpp
 * \author Stuart R. Slattery
 * \brief Epetra AdjointDomain tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_OpaqueWrapper.hpp>

#include <Epetra_Comm.h>
#include <Epetra_SerialComm.h>

#ifdef HAVE_MPI
#include <Epetra_MpiComm.h>
#include <Teuchos_DefaultMpiComm.hpp>
#endif

//---------------------------------------------------------------------------//
// Helper functions.
//---------------------------------------------------------------------------//

Teuchos::RCP<const Epetra_Comm> getEpetraComm()
{
#ifdef HAVE_MPI
    return Teuchos::rcp( new Epetra_MpiComm(MPI_COMM_WORLD) );
#else
    return Teuchos::rcp( new Epetra_SerialComm() );
#endif
}

//---------------------------------------------------------------------------//
Teuchos::RCP<const Teuchos::Comm<int> > getTeuchosCommFromEpetra(
    const Teuchos::RCP<const Epetra_Comm>& epetra_comm )
{
    Teuchos::RCP<const Teuchos::MpiComm<int> > teuchos_comm;

#ifdef HAVE_MPI
	Teuchos::RCP<const Epetra_MpiComm> mpi_epetra_comm =
	    Teuchos::rcp_dynamic_cast<const Epetra_MpiComm>( epetra_comm );

	Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm> >
	    raw_mpi_comm = Teuchos::opaqueWrapper( mpi_epetra_comm->Comm() );

	teuchos_comm =
	    Teuchos::rcp( new Teuchos::MpiComm<int>( raw_mpi_comm ) );
#else
	teuchos_comm = Teuchos::DefaultComm<int>::getComm();
#endif

	return teuchos_comm;
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( DomainCommunicator, Communicate )
{
    // Setup communicators.
    Teuchos::RCP<const Teuchos::Comm<int> > default_comm = 
	Teuchos::DefaultComm<int>::getComm();

    Teuchos::RCP<const Epetra_Comm> epetra_comm = getEpetraComm();

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	getTeuchosCommFromEpetra( epetra_comm );

    TEST_EQUALITY( epetra_comm->MyPID(), comm->getRank() );
    TEST_EQUALITY( epetra_comm->NumProc(), comm->getSize() );

    TEST_EQUALITY( epetra_comm->MyPID(), default_comm->getRank() );
    TEST_EQUALITY( epetra_comm->NumProc(), default_comm->getSize() );

    Teuchos::RCP<const Teuchos::Comm<int> > comm_dup = comm->duplicate();
    TEST_EQUALITY( epetra_comm->MyPID(), comm_dup->getRank() );
    TEST_EQUALITY( epetra_comm->NumProc(), comm_dup->getSize() );

    Teuchos::RCP<const Epetra_Comm> epetra_clone = 
	Teuchos::rcp( getEpetraComm()->Clone() );
    TEST_EQUALITY( epetra_clone->MyPID(), comm->getRank() );
    TEST_EQUALITY( epetra_clone->NumProc(), comm->getSize() );

    Teuchos::RCP<const Teuchos::Comm<int> > comm_clone = 
	getTeuchosCommFromEpetra( epetra_clone );
    TEST_EQUALITY( comm_clone->getRank(), comm_dup->getRank() );
    TEST_EQUALITY( comm_clone->getSize(), comm_dup->getSize() );

    TEST_EQUALITY( comm_clone->getRank(), default_comm->getRank() );
    TEST_EQUALITY( comm_clone->getSize(), default_comm->getSize() );

    // Do blocking communications.


    // Do non-blocking communications.
}

//---------------------------------------------------------------------------//
// end tstEpetraDomainCommunicator.cpp
//---------------------------------------------------------------------------//

