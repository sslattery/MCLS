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
 * \file MCLS_CommTools.hpp
 * \author Stuart R. Slattery
 * \brief CommTools definition.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_COMMTOOLS_HPP
#define MCLS_COMMTOOLS_HPP

#include "MCLS_DBC.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class CommTools
 * \brief Tools for comm distributions.
 */
class CommTools
{
  public:

    /*
     * \brief Given a comm request, check to see if it has completed.
     */
    static bool 
    isRequestComplete( Teuchos::RCP<Teuchos::CommRequest<int> >& handle )
    {
	bool is_complete = false;

#ifdef HAVE_MPI
	Require( !handle.is_null() );
	Teuchos::ArrayView<char>::size_type num_bytes = 
	    Teuchos::rcp_dynamic_cast<Teuchos::MpiCommRequest<int> >( 
		handle )->numBytes();
	MPI_Request raw_request = 
	    Teuchos::rcp_dynamic_cast<Teuchos::MpiCommRequest<int> >( 
		handle )->releaseRawMpiRequest();
	MPI_Status raw_status;
	int flag = 0;
	MPI_Test( &raw_request, &flag, &raw_status );
	is_complete = ( flag != 0 );
	handle = Teuchos::mpiCommRequest<int>( raw_request, num_bytes );
#else
	is_complete = true;
#endif

	return is_complete;
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_COMMTOOLS_HPP

//---------------------------------------------------------------------------//
// end MCLS_CommTools.hpp
// ---------------------------------------------------------------------------//

