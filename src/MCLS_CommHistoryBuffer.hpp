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
 * \file MCLS_CommHistoryBuffer.hpp
 * \author Stuart R. Slattery
 * \brief CommHistoryBuffer class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_COMMHISTORYBUFFER_HPP
#define MCLS_COMMHISTORYBUFFER_HPP

#include "MCLS_DBC.hpp"
#include "MCLS_HistoryBuffer.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class CommHistoryBuffer
 * \brief Data buffer for communicating histories. Tom Evans is responsible
 * for the design of this class and subsequent inheritance structure.
 */
//---------------------------------------------------------------------------//
template<class HT>
class CommHistoryBuffer : public HistoryBuffer<HT>
{
  public:

    //@{
    //! Typedefs.
    typedef HistoryBuffer<HT>                      Base;
    typedef typename Base::history_type            history_type;
    typedef Teuchos::CommRequest<int>              Request;
    typedef Teuchos::Comm<int>                     Comm;
    //@}

  public:

    //! Default constructor.
    CommHistoryBuffer()
	: d_handle( Teuchos::null )
    { Ensure( Base::isEmpty() ); }

    //! Comm constructor.
    CommHistoryBuffer( const Teuchos::RCP<const Comm>& comm_blocking,
		       const Teuchos::RCP<const Comm>& comm_nonblocking )
	: d_handle( Teuchos::null )
	, d_comm_blocking( comm_blocking )
	, d_comm_nonblocking( comm_nonblocking )
    { Ensure( Base::isEmpty() ); }

    //! Size constructor.
    CommHistoryBuffer( const Teuchos::RCP<const Comm>& comm_blocking,
		       const Teuchos::RCP<const Comm>& comm_nonblocking,
		       std::size_t size, int num_history )
	: Base( size, num_history )
	, d_handle( Teuchos::null )
	, d_comm_blocking( comm_blocking )
	, d_comm_nonblocking( comm_nonblocking )
    {
	Ensure( Base::isEmpty() );
	Ensure( Base::allocatedSize() > 0 );
    }

    // Pure virtual destructor.
    virtual ~CommHistoryBuffer() = 0;

    //! Asynchronous post.
    virtual void post( int rank ) = 0;

    //! Asynchronous wait.
    virtual void wait() = 0;

    //! Asynchronous check.
    virtual bool check() = 0;

    //! Free non-blocking communication buffer handles.
    inline void free()
    {
	d_handle = Teuchos::null;
	Base::empty();
	Ensure( Base::isEmpty() );
	Ensure( d_handle.is_null() );
    }

    //! Check the status of a non-blocking communication buffer.
    inline bool status() const
    { return !d_handle.is_null(); }

    //! Set the communicators for this buffer.
    void setComm( const Teuchos::RCP<const Comm>& comm_blocking,
		  const Teuchos::RCP<const Comm>& comm_nonblocking )
    { 
	d_comm_blocking = comm_blocking; 
	d_comm_nonblocking = comm_nonblocking; 
	Ensure( !d_comm_blocking.is_null() );
	Ensure( !d_comm_nonblocking.is_null() );
    }

  protected:

    // Non-blocking communication handles. This object's destructor will
    // cancel the request. A handle is in use if it is non-null.
    Teuchos::RCP<Request> d_handle;

    // Communicator on which blocking communications for this buffer is
    // defined.
    Teuchos::RCP<const Comm> d_comm_blocking;

    // Communicator on which nonblocking communications for this buffer is
    // defined.
    Teuchos::RCP<const Comm> d_comm_nonblocking;
};

//---------------------------------------------------------------------------//
/*!
 * \class ReceiveHistoryBuffer
 * \brief Data buffer for receiving histories. Tom Evans is responsible for
 * the design of this class and subsequent inheritance structure.
 */
//---------------------------------------------------------------------------//
template<class HT>
class ReceiveHistoryBuffer : public CommHistoryBuffer<HT>
{
  public:

    //@{
    //! Typedefs.
    typedef HistoryBuffer<HT>                      Root;
    typedef CommHistoryBuffer<HT>                  Base;
    typedef typename Base::Comm                    Comm;
    //@}

  public:

    //! Default constructor.
    ReceiveHistoryBuffer()
    { Ensure( Base::isEmpty() ); }

    //! Comm constructor.
    ReceiveHistoryBuffer( const Teuchos::RCP<const Comm>& comm_blocking,
			  const Teuchos::RCP<const Comm>& comm_nonblocking )
	: Base( comm_blocking, comm_nonblocking )
    { Ensure( Base::isEmpty() ); }

    //! Size constructor.
    ReceiveHistoryBuffer( const Teuchos::RCP<const Comm>& comm_blocking,
			  const Teuchos::RCP<const Comm>& comm_nonblocking,
			  std::size_t size, int num_history )
	: Base( comm_blocking, comm_nonblocking, size, num_history )
    {
	Ensure( Base::isEmpty() );
	Ensure( Base::allocatedSize() > 0 );
    }

    //! Destructor.
    ~ReceiveHistoryBuffer()
    { /* ... */ }

    // Blocking receive.
    void receive( int rank );

    // Asynchronous post.
    void post( int rank );

    // Asynchronous wait.
    void wait();

    // Asynchronous check.
    bool check();
};

//---------------------------------------------------------------------------//
/*!
 * \class SendHistoryBuffer
 * \brief Data buffer for sending histories. Tom Evans is responsible for the
 * design of this class and subsequent inheritance structure.
 */
//---------------------------------------------------------------------------//
template<class HT>
class SendHistoryBuffer : public CommHistoryBuffer<HT>
{
  public:

    //@{
    //! Typedefs.
    typedef HistoryBuffer<HT>                      Root;
    typedef CommHistoryBuffer<HT>                  Base;
    typedef typename Base::Comm                    Comm;
    //@}

  public:

    //! Default constructor.
    SendHistoryBuffer()
    { Ensure( Base::isEmpty() ); }

    //! Comm constructor.
    SendHistoryBuffer( const Teuchos::RCP<const Comm>& comm_blocking,
		       const Teuchos::RCP<const Comm>& comm_nonblocking )
	: Base( comm_blocking, comm_nonblocking )
    { Ensure( Base::isEmpty() ); }

    //! Size constructor.
    SendHistoryBuffer( const Teuchos::RCP<const Comm>& comm_blocking,
		       const Teuchos::RCP<const Comm>& comm_nonblocking,
		       std::size_t size, int num_history )
	: Base( comm_blocking, comm_nonblocking, size, num_history )
    {
	Ensure( Base::isEmpty() );
	Ensure( Base::allocatedSize() > 0 );
    }

    //! Destructor.
    ~SendHistoryBuffer()
    { /* ... */ }

    // Blocking send.
    void send( int rank );

    // Asynchronous post.
    void post( int rank );

    // Asynchronous wait.
    void wait();

    // Asynchronous check.
    bool check();
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_CommHistoryBuffer_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_COMMHISTORYBUFFER_HPP

//---------------------------------------------------------------------------//
// end MCLS_CommHistoryBuffer.hpp
//---------------------------------------------------------------------------//

