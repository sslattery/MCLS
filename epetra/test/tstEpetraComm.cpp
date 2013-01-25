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
#include <Teuchos_Ptr.hpp>

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
    int comm_rank = default_comm->getRank();

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

    Teuchos::RCP<const Teuchos::Comm<int> > clone_dup = comm_clone->duplicate();
    TEST_EQUALITY( clone_dup->getRank(), default_comm->getRank() );
    TEST_EQUALITY( clone_dup->getSize(), default_comm->getSize() );

    // Barrier before proceeding.
    default_comm->barrier();

    // Do blocking communications.
    if ( comm_rank == 0 )
    {
	int send_val_1 = 5;
	int send_val_2 = 4;
	int send_val_3 = 3;
	int send_val_4 = 2;
	int send_val_5 = 1;

	int receive_val_1 = 0;
	int receive_val_2 = 0;
	int receive_val_3 = 0;
	int receive_val_4 = 0;
	int receive_val_5 = 0;

	// Send to proc 1 for each comm.
	Teuchos::send<int,int>( *default_comm, send_val_1, 1 );
	Teuchos::send<int,int>( *comm, send_val_2, 1 );
	Teuchos::send<int,int>( *comm_dup, send_val_3, 1 );
	Teuchos::send<int,int>( *comm_clone, send_val_4, 1 );
	Teuchos::send<int,int>( *clone_dup, send_val_5, 1 );

	// Receive from proc 1 for each comm.
	Teuchos::receive<int,int>( *default_comm, 1, &receive_val_1 );
	Teuchos::receive<int,int>( *comm, 1, &receive_val_2 );
	Teuchos::receive<int,int>( *comm_dup, 1, &receive_val_3 );
	Teuchos::receive<int,int>( *comm_clone, 1, &receive_val_4 );
	Teuchos::receive<int,int>( *clone_dup, 1, &receive_val_5 );

	// Check the messaging.
	TEST_EQUALITY( receive_val_1, send_val_1 );
	TEST_EQUALITY( receive_val_2, send_val_2 );
	TEST_EQUALITY( receive_val_3, send_val_3 );
	TEST_EQUALITY( receive_val_4, send_val_4 );
	TEST_EQUALITY( receive_val_5, send_val_5 );
    }
    else if ( comm_rank == 1 )
    {
	int send_val_1 = 5;
	int send_val_2 = 4;
	int send_val_3 = 3;
	int send_val_4 = 2;
	int send_val_5 = 1;

	int receive_val_1 = 0;
	int receive_val_2 = 0;
	int receive_val_3 = 0;
	int receive_val_4 = 0;
	int receive_val_5 = 0;

	// Receive from proc 0 for each comm.
	Teuchos::receive<int,int>( *default_comm, 0, &receive_val_1 );
	Teuchos::receive<int,int>( *comm, 0, &receive_val_2 );
	Teuchos::receive<int,int>( *comm_dup, 0, &receive_val_3 );
	Teuchos::receive<int,int>( *comm_clone, 0, &receive_val_4 );
	Teuchos::receive<int,int>( *clone_dup, 0, &receive_val_5 );

	// Send to proc 0 for each comm.
	Teuchos::send<int,int>( *default_comm, send_val_1, 0 );
	Teuchos::send<int,int>( *comm, send_val_2, 0 );
	Teuchos::send<int,int>( *comm_dup, send_val_3, 0 );
	Teuchos::send<int,int>( *comm_clone, send_val_4, 0 );
	Teuchos::send<int,int>( *clone_dup, send_val_5, 0 );

	// Check the messaging.
	TEST_EQUALITY( receive_val_1, send_val_1 );
	TEST_EQUALITY( receive_val_2, send_val_2 );
	TEST_EQUALITY( receive_val_3, send_val_3 );
	TEST_EQUALITY( receive_val_4, send_val_4 );
	TEST_EQUALITY( receive_val_5, send_val_5 );
    }

    // Do non-blocking communications.
    if ( comm_rank == 0 )
    {
	Teuchos::RCP<Teuchos::CommRequest<int> > send_r_1;
	Teuchos::RCP<Teuchos::CommRequest<int> > send_r_2;
	Teuchos::RCP<Teuchos::CommRequest<int> > send_r_3;
	Teuchos::RCP<Teuchos::CommRequest<int> > send_r_4;
	Teuchos::RCP<Teuchos::CommRequest<int> > send_r_5;

	Teuchos::RCP<Teuchos::CommRequest<int> > receive_r_1;
	Teuchos::RCP<Teuchos::CommRequest<int> > receive_r_2;
	Teuchos::RCP<Teuchos::CommRequest<int> > receive_r_3;
	Teuchos::RCP<Teuchos::CommRequest<int> > receive_r_4;
	Teuchos::RCP<Teuchos::CommRequest<int> > receive_r_5;

	Teuchos::RCP<int> send_val_1 = Teuchos::rcp( new int(5) );
	Teuchos::RCP<int> send_val_2 = Teuchos::rcp( new int(4) );
	Teuchos::RCP<int> send_val_3 = Teuchos::rcp( new int(3) );
	Teuchos::RCP<int> send_val_4 = Teuchos::rcp( new int(2) );
	Teuchos::RCP<int> send_val_5 = Teuchos::rcp( new int(1) );

	Teuchos::RCP<int> receive_val_1 = Teuchos::rcp( new int(0) );
	Teuchos::RCP<int> receive_val_2 = Teuchos::rcp( new int(0) );
	Teuchos::RCP<int> receive_val_3 = Teuchos::rcp( new int(0) );
	Teuchos::RCP<int> receive_val_4 = Teuchos::rcp( new int(0) );
	Teuchos::RCP<int> receive_val_5 = Teuchos::rcp( new int(0) );

	// Post receives.
	receive_r_1 = Teuchos::ireceive<int,int>( *default_comm, receive_val_1, 1 );
	receive_r_2 = Teuchos::ireceive<int,int>( *comm,         receive_val_2, 1 );
	receive_r_3 = Teuchos::ireceive<int,int>( *comm_dup,     receive_val_3, 1 );
	receive_r_4 = Teuchos::ireceive<int,int>( *comm_clone,   receive_val_4, 1 );
	receive_r_5 = Teuchos::ireceive<int,int>( *clone_dup,    receive_val_5, 1 );

	// Post sends.
	send_r_1 = Teuchos::isend<int,int>( *default_comm, send_val_1, 1 );
	send_r_2 = Teuchos::isend<int,int>( *comm,         send_val_2, 1 );
	send_r_3 = Teuchos::isend<int,int>( *comm_dup,     send_val_3, 1 );
	send_r_4 = Teuchos::isend<int,int>( *comm_clone,   send_val_4, 1 );
	send_r_5 = Teuchos::isend<int,int>( *clone_dup,    send_val_5, 1 );

	// Wait on sends.
	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    send_1_ptr( &send_r_1 );
	Teuchos::wait( *default_comm, send_1_ptr );
	TEST_ASSERT( send_r_1.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    send_2_ptr( &send_r_2 );
	Teuchos::wait( *comm, send_2_ptr );
	TEST_ASSERT( send_r_2.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    send_3_ptr( &send_r_3 );
	Teuchos::wait( *comm_dup, send_3_ptr );
	TEST_ASSERT( send_r_3.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    send_4_ptr( &send_r_4 );
	Teuchos::wait( *comm_clone, send_4_ptr );
	TEST_ASSERT( send_r_4.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    send_5_ptr( &send_r_5 );
	Teuchos::wait( *clone_dup, send_5_ptr );
	TEST_ASSERT( send_r_5.is_null() );

	// Wait on the receives.
	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    receive_1_ptr( &receive_r_1 );
	Teuchos::wait( *default_comm, receive_1_ptr );
	TEST_ASSERT( receive_r_1.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    receive_2_ptr( &receive_r_2 );
	Teuchos::wait( *comm, receive_2_ptr );
	TEST_ASSERT( receive_r_2.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    receive_3_ptr( &receive_r_3 );
	Teuchos::wait( *comm_dup, receive_3_ptr );
	TEST_ASSERT( receive_r_3.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    receive_4_ptr( &receive_r_4 );
	Teuchos::wait( *comm_clone, receive_4_ptr );
	TEST_ASSERT( receive_r_4.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    receive_5_ptr( &receive_r_5 );
	Teuchos::wait( *clone_dup, receive_5_ptr );
	TEST_ASSERT( receive_r_5.is_null() );

	// Check the messaging.
	TEST_EQUALITY( *receive_val_1, *send_val_1 );
	TEST_EQUALITY( *receive_val_2, *send_val_2 );
	TEST_EQUALITY( *receive_val_3, *send_val_3 );
	TEST_EQUALITY( *receive_val_4, *send_val_4 );
	TEST_EQUALITY( *receive_val_5, *send_val_5 );
    }
    else if ( comm_rank == 1 )
    {
	Teuchos::RCP<Teuchos::CommRequest<int> > send_r_1;
	Teuchos::RCP<Teuchos::CommRequest<int> > send_r_2;
	Teuchos::RCP<Teuchos::CommRequest<int> > send_r_3;
	Teuchos::RCP<Teuchos::CommRequest<int> > send_r_4;
	Teuchos::RCP<Teuchos::CommRequest<int> > send_r_5;

	Teuchos::RCP<Teuchos::CommRequest<int> > receive_r_1;
	Teuchos::RCP<Teuchos::CommRequest<int> > receive_r_2;
	Teuchos::RCP<Teuchos::CommRequest<int> > receive_r_3;
	Teuchos::RCP<Teuchos::CommRequest<int> > receive_r_4;
	Teuchos::RCP<Teuchos::CommRequest<int> > receive_r_5;

	Teuchos::RCP<int> send_val_1 = Teuchos::rcp( new int(5) );
	Teuchos::RCP<int> send_val_2 = Teuchos::rcp( new int(4) );
	Teuchos::RCP<int> send_val_3 = Teuchos::rcp( new int(3) );
	Teuchos::RCP<int> send_val_4 = Teuchos::rcp( new int(2) );
	Teuchos::RCP<int> send_val_5 = Teuchos::rcp( new int(1) );

	Teuchos::RCP<int> receive_val_1 = Teuchos::rcp( new int(0) );
	Teuchos::RCP<int> receive_val_2 = Teuchos::rcp( new int(0) );
	Teuchos::RCP<int> receive_val_3 = Teuchos::rcp( new int(0) );
	Teuchos::RCP<int> receive_val_4 = Teuchos::rcp( new int(0) );
	Teuchos::RCP<int> receive_val_5 = Teuchos::rcp( new int(0) );

	// Post receives.
	receive_r_1 = Teuchos::ireceive<int,int>( *default_comm, receive_val_1, 0 );
	receive_r_2 = Teuchos::ireceive<int,int>( *comm,         receive_val_2, 0 );
	receive_r_3 = Teuchos::ireceive<int,int>( *comm_dup,     receive_val_3, 0 );
	receive_r_4 = Teuchos::ireceive<int,int>( *comm_clone,   receive_val_4, 0 );
	receive_r_5 = Teuchos::ireceive<int,int>( *clone_dup,    receive_val_5, 0 );

	// Post sends.
	send_r_1 = Teuchos::isend<int,int>( *default_comm, send_val_1, 0 );
	send_r_2 = Teuchos::isend<int,int>( *comm,         send_val_2, 0 );
	send_r_3 = Teuchos::isend<int,int>( *comm_dup,     send_val_3, 0 );
	send_r_4 = Teuchos::isend<int,int>( *comm_clone,   send_val_4, 0 );
	send_r_5 = Teuchos::isend<int,int>( *clone_dup,    send_val_5, 0 );

	// Wait on sends.
	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    send_1_ptr( &send_r_1 );
	Teuchos::wait( *default_comm, send_1_ptr );
	TEST_ASSERT( send_r_1.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    send_2_ptr( &send_r_2 );
	Teuchos::wait( *comm, send_2_ptr );
	TEST_ASSERT( send_r_2.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    send_3_ptr( &send_r_3 );
	Teuchos::wait( *comm_dup, send_3_ptr );
	TEST_ASSERT( send_r_3.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    send_4_ptr( &send_r_4 );
	Teuchos::wait( *comm_clone, send_4_ptr );
	TEST_ASSERT( send_r_4.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    send_5_ptr( &send_r_5 );
	Teuchos::wait( *clone_dup, send_5_ptr );
	TEST_ASSERT( send_r_5.is_null() );

	// Wait on the receives.
	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    receive_1_ptr( &receive_r_1 );
	Teuchos::wait( *default_comm, receive_1_ptr );
	TEST_ASSERT( receive_r_1.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    receive_2_ptr( &receive_r_2 );
	Teuchos::wait( *comm, receive_2_ptr );
	TEST_ASSERT( receive_r_2.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    receive_3_ptr( &receive_r_3 );
	Teuchos::wait( *comm_dup, receive_3_ptr );
	TEST_ASSERT( receive_r_3.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    receive_4_ptr( &receive_r_4 );
	Teuchos::wait( *comm_clone, receive_4_ptr );
	TEST_ASSERT( receive_r_4.is_null() );

	Teuchos::Ptr<Teuchos::RCP<Teuchos::CommRequest<int> > >
	    receive_5_ptr( &receive_r_5 );
	Teuchos::wait( *clone_dup, receive_5_ptr );
	TEST_ASSERT( receive_r_5.is_null() );

	// Check the messaging.
	TEST_EQUALITY( *receive_val_1, *send_val_1 );
	TEST_EQUALITY( *receive_val_2, *send_val_2 );
	TEST_EQUALITY( *receive_val_3, *send_val_3 );
	TEST_EQUALITY( *receive_val_4, *send_val_4 );
	TEST_EQUALITY( *receive_val_5, *send_val_5 );
    }
}

//---------------------------------------------------------------------------//
// end tstEpetraDomainCommunicator.cpp
//---------------------------------------------------------------------------//

