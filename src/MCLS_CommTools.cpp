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
 * \file MCLS_CommTools.cpp
 * \author Stuart R. Slattery
 * \brief CommTools implementation.
 */
//---------------------------------------------------------------------------//

#include "MCLS_CommTools.hpp"
#include "MCLS_DBC.hpp"

#ifdef HAVE_MPI
#include <Teuchos_DefaultMpiComm.hpp>
#endif

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Given a comm request, check to see if it has completed.
 */
bool 
CommTools::isRequestComplete( Teuchos::RCP<Teuchos::CommRequest<int> >& handle )
{
    bool is_complete = false;

#ifdef HAVE_MPI
    MCLS_REQUIRE( Teuchos::nonnull(handle) );
    Teuchos::RCP<Teuchos::MpiCommRequestBase<int> > handle_base =
	Teuchos::rcp_dynamic_cast<Teuchos::MpiCommRequestBase<int> >(handle);
    MCLS_CHECK( Teuchos::nonnull(handle_base) );
    MPI_Request raw_request = handle_base->releaseRawMpiRequest();
    MPI_Status raw_status;
    int flag = 0;
    MPI_Test( &raw_request, &flag, &raw_status );
    is_complete = ( flag != 0 );
    handle = Teuchos::rcp( 
	new Teuchos::MpiCommRequestBase<int>(raw_request) );
#else
    is_complete = true;
#endif

    return is_complete;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Do a reduce sum for a given buffer.
 *
 * Float instantiation.
 */
template<>
void CommTools::reduceSum<float>( 
    const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
    const int root,
    const int count,
    const float send_buffer[],
    float global_reducts[] )
{
#ifdef HAVE_MPI
    const Teuchos::RCP<const Teuchos::MpiComm<int> > mpi_comm =
	Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int> >( comm );
    MPI_Comm raw_mpi_comm = *( mpi_comm->getRawMpiComm() );
    const int error = MPI_Reduce( 
        const_cast<float*>(send_buffer),
	global_reducts,
	count,
	MPI_FLOAT,
	MPI_SUM,
	root,
	raw_mpi_comm );
    MCLS_INSIST( MPI_SUCCESS == error, "Reduce Sum Failed" );
#else
    std::copy( send_buffer, send_buffer+count, global_reducts );
#endif
}

//---------------------------------------------------------------------------//
/*!
 * \brief Do a reduce sum for a given buffer.
 *
 * Double instantiation.
 */
template<>
void CommTools::reduceSum<double>( 
    const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
    const int root,
    const int count,
    const double send_buffer[],
    double global_reducts[] )
{
#ifdef HAVE_MPI
    const Teuchos::RCP<const Teuchos::MpiComm<int> > mpi_comm =
	Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int> >( comm );
    MPI_Comm raw_mpi_comm = *( mpi_comm->getRawMpiComm() );
    const int error = MPI_Reduce( 
        const_cast<double*>(send_buffer),
	global_reducts,
	count,
	MPI_DOUBLE,
	MPI_SUM,
	root,
	raw_mpi_comm );
    MCLS_CHECK( MPI_SUCCESS == error );
#else
    std::copy( send_buffer, send_buffer+count, global_reducts );
#endif
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_CommTools.cpp
// ---------------------------------------------------------------------------//

