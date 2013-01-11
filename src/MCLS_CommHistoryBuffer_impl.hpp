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
 * \brief Data buffer for histories. Tom Evans is responsible for the design
 * of this class and subsequent inheritance structure.
 */
//---------------------------------------------------------------------------//
template<class HT>
class CommHistoryBuffer
{
  public:

    //@{
    //! Typedefs.
    typedef HistoryBuffer<HT>                      Base;
    typedef typename Base::history_type            history_type;
    typedef typename Base::Buffer                  Buffer;
    //@}

  public:

    //! Default constructor.
    CommHistoryBuffer()
    { Ensure( Base::isEmpty() ); }

    //! Size constructor.
    CommHistoryBuffer( std::size_t size, int num_history )
	: Base( size, num_history )
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
    }

    //! Check the status of a non-blocking communication buffer.
    inline bool status() const
    { 
	Require( !d_handle.is_null() );
	return ( d_handle->getSourceRank() >= 0 ); 
    }

  protected:

    // Non-blocking communication handles. This object's destructor will
    // cancel the request.
    Teuchos::RCP<Teuchos::CommRequest<int> > d_handle;
};

//---------------------------------------------------------------------------//
/*!
 * \class ReceiveCommHistoryBuffer
 * \brief Data buffer for receiving histories. Tom Evans is responsible for
 * the design of this class and subsequent inheritance structure.
 */
//---------------------------------------------------------------------------//
template<class HT>
class ReceiveCommHistoryBuffer : public CommHistoryBuffer<HT>
{
  public:

    //@{
    //! Typedefs.
    typedef HistoryBuffer<HT>                      Root;
    typedef CommHistoryBuffer<HT>                  Base;
    typedef typename Base::history_type            history_type;
    typedef typename Base::Buffer                  Buffer;
    //@}

  public:

    //! Default constructor.
    ReceiveCommHistoryBuffer()
    { Ensure( Base::isEmpty() ); }

    //! Size constructor.
    ReceiveCommHistoryBuffer( std::size_t size, int num_history )
	: Base( size, num_history )
    {
	Ensure( Base::isEmpty() );
	Ensure( Base::allocatedSize() > 0 );
    }

    //! Destructor.
    ~ReceiveCommHistoryBuffer()
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
 * \class SendCommHistoryBuffer
 * \brief Data buffer for sending histories. Tom Evans is responsible for the
 * design of this class and subsequent inheritance structure.
 */
//---------------------------------------------------------------------------//
template<class HT>
class SendCommHistoryBuffer : public CommHistoryBuffer<HT>
{
  public:

    //@{
    //! Typedefs.
    typedef HistoryBuffer<HT>                      Root;
    typedef CommHistoryBuffer<HT>                  Base;
    typedef typename Base::history_type            history_type;
    typedef typename Base::Buffer                  Buffer;
    //@}

  public:

    //! Default constructor.
    SendCommHistoryBuffer()
    { Ensure( Base::isEmpty() ); }

    //! Size constructor.
    SendCommHistoryBuffer( std::size_t size, int num_history )
	: Base( size, num_history )
    {
	Ensure( Base::isEmpty() );
	Ensure( Base::allocatedSize() > 0 );
    }

    //! Destructor.
    ~SendCommHistoryBuffer()
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

