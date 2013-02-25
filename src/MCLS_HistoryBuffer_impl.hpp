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
 * \file MCLS_HistoryBuffer_impl.hpp
 * \author Stuart R. Slattery
 * \brief HistoryBuffer class implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_HISTORYBUFFER_IMPL_HPP
#define MCLS_HISTORYBUFFER_IMPL_HPP

#include <algorithm>

#include "MCLS_DBC.hpp"
#include "MCLS_Serializer.hpp"

#include <Teuchos_as.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Size constructor.
 */
template<class HT>
HistoryBuffer<HT>::HistoryBuffer( std::size_t size, int num_history )
    : d_number( 0 )
{
    setSizePackedHistory( size );
    setMaxNumHistories( num_history );
    allocate();
    MCLS_ENSURE( !d_buffer.empty() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Allocate the buffer for the byte size of the maximum number of
 * histories plus an additional integer for the actual number of the buffer.
 */
template<class HT>
void HistoryBuffer<HT>::allocate()
{
    MCLS_REQUIRE( d_number == 0 );
    d_buffer.resize( 
	d_max_num_histories*d_size_packed_history + sizeof(int), '\0' );
    MCLS_ENSURE( d_number == 0 );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Deallocate the buffer.
 */
template<class HT>
void HistoryBuffer<HT>::deallocate()
{
    MCLS_REQUIRE( d_number == 0 );
    d_buffer.clear();
    MCLS_ENSURE( d_number == 0 );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Write a history into the buffer.
 */
template<class HT>
void HistoryBuffer<HT>::bufferHistory( const HT& history )
{
    MCLS_REQUIRE( d_size_packed_history > 0 );
    MCLS_REQUIRE( d_number < d_max_num_histories );
    MCLS_REQUIRE( d_number >= 0 );
    MCLS_REQUIRE( !d_buffer.empty() );

    Buffer packed_history = history.pack();
    MCLS_CHECK( Teuchos::as<std::size_t>(packed_history.size()) == 
	   d_size_packed_history );

    Buffer::iterator buffer_it = d_buffer.begin() + 
				 d_size_packed_history*d_number;
    MCLS_REQUIRE( buffer_it != d_buffer.end() );

    std::copy( packed_history.begin(), packed_history.end(), buffer_it );
    ++d_number;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Add the histories in the buffer to a bank.
 */
template<class HT>
void HistoryBuffer<HT>::addToBank( BankType& bank )
{
    MCLS_REQUIRE( d_size_packed_history > 0 );

    Buffer::const_iterator buffer_it = d_buffer.begin();
    Buffer packed_history( d_size_packed_history );
    Teuchos::RCP<HT> history;

    MCLS_REMEMBER( std::size_t bank_size = bank.size() );

    for ( int n = 0; n < d_number; ++n )
    {
	std::copy( buffer_it, buffer_it + d_size_packed_history, 
		   packed_history.begin() );

	history = Teuchos::rcp( new HT(packed_history) );
	MCLS_CHECK( !history.is_null() );
	bank.push( history );

	buffer_it += d_size_packed_history;
    }

    MCLS_ENSURE( bank_size + d_number == bank.size() );
    MCLS_ENSURE( d_number == d_max_num_histories ?
            buffer_it + sizeof(int) == d_buffer.end() :
            buffer_it + sizeof(int) != d_buffer.end() );

    empty();
    MCLS_ENSURE( isEmpty() );
}

//---------------------------------------------------------------------------//
// Protected Members.
//---------------------------------------------------------------------------//
/*!
 * \brief Add the number of histories to the end of the buffer.
 */
template<class HT>
void HistoryBuffer<HT>::writeNumToBuffer()
{
    MCLS_REQUIRE( Teuchos::as<std::size_t>(d_buffer.size()) > sizeof(int) );
    Serializer s;
    s.setBuffer( sizeof(int), &d_buffer[d_buffer.size() - sizeof(int)] );
    s << d_number;
    MCLS_ENSURE( s.getPtr() == &d_buffer[0] + d_buffer.size() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Read the number of histories from the end of the buffer.
 */
template<class HT>
void HistoryBuffer<HT>::readNumFromBuffer()
{
    Deserializer ds;
    ds.setBuffer( sizeof(int), &d_buffer[d_buffer.size() - sizeof(int)] );
    ds >> d_number;
    MCLS_ENSURE( ds.getPtr() == &d_buffer[0] + d_buffer.size() );
    MCLS_ENSURE( d_number >= 0 );
}

//---------------------------------------------------------------------------//
// Static Members.
//---------------------------------------------------------------------------//

//! Default maximum number of history allowed in a buffer.
template<class HT>
int HistoryBuffer<HT>::d_max_num_histories = 1000;

//! Default size of a packed history.
template<class HT>
std::size_t HistoryBuffer<HT>::d_size_packed_history = 0;

//---------------------------------------------------------------------------//
/*!
 * \brief Set the maximum number of histories allowed in the buffer.
 */
template<class HT>
void HistoryBuffer<HT>::setMaxNumHistories( int num_history )
{
    MCLS_REQUIRE( num_history > 0 );
    MCLS_REQUIRE( d_size_packed_history > 0 );
    d_max_num_histories = num_history;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the byte size of a packed history.
 */
template<class HT>
void HistoryBuffer<HT>::setSizePackedHistory( std::size_t size )
{
    MCLS_REQUIRE( size > 0 );
    d_size_packed_history = size;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_HISTORYBUFFER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_HistoryBuffer_impl.hpp
//---------------------------------------------------------------------------//

