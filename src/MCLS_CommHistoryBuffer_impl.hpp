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
 * \file MCLS_CommHistoryBuffer_impl.hpp
 * \author Stuart R. Slattery
 * \brief CommHistoryBuffer class implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_COMMHISTORYBUFFER_IMPL_HPP
#define MCLS_COMMHISTORYBUFFER_IMPL_HPP

#include <MCLS_CommTools.hpp>

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_Ptr.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
// CommHistoryBuffer functions.
//---------------------------------------------------------------------------//
/*!
 * \brief Pure virtual destructor. Prevents direct instantiation of
 * CommHistoryBuffer.
 */
template<class History>
CommHistoryBuffer<History>::~CommHistoryBuffer()
{ /* ... */ }

//---------------------------------------------------------------------------//
// ReceiveHistoryBuffer functions.
//---------------------------------------------------------------------------//
/*!
 * \brief Blocking receive.
 */
template<class History>
void ReceiveHistoryBuffer<History>::receive( int rank )
{
    MCLS_REQUIRE( Teuchos::nonnull(Base::d_comm_blocking) );
    MCLS_REQUIRE( Root::isEmpty() );
    MCLS_REQUIRE( Root::allocatedSize() > sizeof(int) );

    Teuchos::receive<int,char>( 
	*Base::d_comm_blocking, rank, 
	Root::d_buffer.size(), Root::d_buffer.getRawPtr() );
    Root::readNumFromBuffer();

    MCLS_ENSURE( Root::d_number < Root::maxNum() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Post non-blocking receives.
 */
template<class History>
void ReceiveHistoryBuffer<History>::post( int rank )
{
    MCLS_REQUIRE( Teuchos::nonnull(Base::d_comm_nonblocking) );
    MCLS_REQUIRE( Root::isEmpty() );
    MCLS_REQUIRE( Root::allocatedSize() > sizeof(int) );

    Base::d_handle = Teuchos::ireceive<int,char>( 
	*Base::d_comm_nonblocking, 
	Teuchos::arcpFromArray(Root::d_buffer), 
	rank );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Wait on a non-blocking receive to finish.
 */
template<class History>
void ReceiveHistoryBuffer<History>::wait()
{
    MCLS_REQUIRE( Teuchos::nonnull(Base::d_comm_nonblocking) );

    Teuchos::Ptr<Teuchos::RCP<typename Base::Request> > 
	request_ptr( &this->d_handle );
    Teuchos::wait( *Base::d_comm_nonblocking, request_ptr );
    Root::readNumFromBuffer();

    MCLS_ENSURE( Teuchos::is_null(Base::d_handle) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Check to see if a non-blocking send has finished.
 */
template<class History>
bool ReceiveHistoryBuffer<History>::check()
{
    MCLS_REQUIRE( Teuchos::nonnull(Base::d_comm_nonblocking) );

    if ( CommTools::isRequestComplete(Base::d_handle) )
    {
	Root::readNumFromBuffer();
	Base::d_handle = Teuchos::null;

	MCLS_ENSURE( Teuchos::is_null(Base::d_handle) );
	MCLS_ENSURE( Root::numHistories() >= 0 );
	return true;
    }

    return false;
}

//---------------------------------------------------------------------------//
// SendHistoryBuffer functions.
//---------------------------------------------------------------------------//
/*!
 * \brief Blocking send.
 */
template<class History>
void SendHistoryBuffer<History>::send( int rank )
{
    MCLS_REQUIRE( Teuchos::nonnull(Base::d_comm_blocking) );
    MCLS_REQUIRE( Root::allocatedSize() > sizeof(int) );

    Root::writeNumToBuffer();
    Teuchos::send<int,char>( *Base::d_comm_blocking, Root::d_buffer.size(), 
			     Root::d_buffer.getRawPtr(), rank );

    Root::empty();

    MCLS_ENSURE( Root::isEmpty() );
    MCLS_ENSURE( Root::allocatedSize() > sizeof(int) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Post non-blocking send.
 */
template<class History>
void SendHistoryBuffer<History>::post( int rank )
{
    MCLS_REQUIRE( Teuchos::nonnull(Base::d_comm_nonblocking) );
    MCLS_REQUIRE( Root::allocatedSize() > sizeof(int) );

    Root::writeNumToBuffer();
    Base::d_handle = Teuchos::isend<int,char>( 
	*Base::d_comm_nonblocking, 
	Teuchos::arcpFromArray(Root::d_buffer), 
	rank );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Wait on a non-blocking send to finish.
 */
template<class History>
void SendHistoryBuffer<History>::wait()
{
    MCLS_REQUIRE( Teuchos::nonnull(Base::d_comm_nonblocking) );

    Teuchos::Ptr<Teuchos::RCP<typename Base::Request> > 
	request_ptr( &this->d_handle );

    Teuchos::wait( *Base::d_comm_nonblocking, request_ptr );

    Root::empty();

    MCLS_ENSURE( Teuchos::is_null(Base::d_handle) );
    MCLS_ENSURE( Root::isEmpty() );
    MCLS_ENSURE( Root::allocatedSize() > sizeof(int) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Check to see if a non-blocking send has finished.
 */
template<class History>
bool SendHistoryBuffer<History>::check()
{
    MCLS_REQUIRE( Teuchos::nonnull(Base::d_comm_nonblocking) );

    if ( CommTools::isRequestComplete(Base::d_handle) )
    {
	Root::empty();
	Base::d_handle = Teuchos::null;

	MCLS_ENSURE( Teuchos::is_null(Base::d_handle) );
	MCLS_ENSURE( Root::isEmpty() );
	return true;
    }

    return false;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_COMMHISTORYBUFFER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_CommHistoryBuffer_impl.hpp
//---------------------------------------------------------------------------//

