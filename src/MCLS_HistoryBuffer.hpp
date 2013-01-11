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
 * \file MCLS_HistoryBuffer.hpp
 * \author Stuart R. Slattery
 * \brief HistoryBuffer class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_HISTORYBUFFER_HPP
#define MCLS_HISTORYBUFFER_HPP

#include <stack>

#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class HistoryBuffer
 * \brief Data buffer for histories. Tom Evans is responsible for the design
 * of this class and subsequent inheritance structure.
 */
//---------------------------------------------------------------------------//
template<class HT>
class HistoryBuffer
{
  public:

    //@{
    //! Typedefs.
    typedef HT                                  history_type;
    typedef Teuchos::Array<char>                Buffer;
    //@}

    //! Default constructor.
    HistoryBuffer()
	: d_number( 0 )
    { /* ... */ }

    // Size constructor.
    HistoryBuffer( std::size_t size, int num_history );

    //! Destructor.
    virtual ~HistoryBuffer()
    { /* ... */ }

    //! Set the number of histories in the buffer to zero.
    void empty()
    { d_number = 0; }

    // Allocate the buffer.
    void allocate();

    // Deallocate the buffer.
    void deallocate();

    // Write a history into the buffer.
    void bufferHistory( const HT& history );

    // Add the histories in the buffer to a bank.
    void addToBank( std::stack<Teuchos::RCP<HT> >& bank );

    //! Get current number of histories in the buffer.
    int numHistories() const
    { return d_number; }

    //! Check if the buffer is empty.
    bool isEmpty() const
    { return ( d_number == 0 ); }

    //! Check if the buffer is full.
    bool isFull() const 
    { return ( d_number == d_max_num_histories ); }

    //! Get the current allocated size of the buffer.
    std::size_t allocatedSize() const
    { return d_buffer.size(); }

  public:

    // Set the maximum number of histories allowed in the buffer.
    static void setMaxNumHistories( int num_history );

    // Set the byte size of a packed history.
    static void setSizePackedHistory( std::size_t size );

    //! Get the maximum number of histories allowed in the buffer.
    static int maxNum()
    { return d_max_num_histories; }

    //! Get the size of a packed history.
    static int sizePackedHistory()
    { return d_size_packed_history; }

  protected:

    // Add the number of histories to the end of the buffer.
    void writeNumToBuffer();

    // Read the number of histories from the end of the buffer.
    void readNumFromBuffer();

  protected:

    // Packed history buffer.
    Buffer d_buffer;

    // Number of histories currently in the buffer.
    int d_number;

  private:

    // Maximum number of histories allowed in the buffer.
    static int d_max_num_histories;

    // Size of a packed history in bytes.
    static std::size_t d_size_packed_history; 
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_HistoryBuffer_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_HISTORYBUFFER_HPP

//---------------------------------------------------------------------------//
// end MCLS_HistoryBuffer.hpp
//---------------------------------------------------------------------------//

