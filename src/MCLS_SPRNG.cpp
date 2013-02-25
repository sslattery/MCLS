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
 * \file MCLS_SPRNG.cpp
 * \author Stuart R. Slattery
 * \brief SPRNG wrapper class implementation.
 */
//---------------------------------------------------------------------------//

#include "MCLS_SPRNG.hpp"
#include "MCLS_Serializer.hpp"

#include <Teuchos_as.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
// STATIC MEMBERS
//---------------------------------------------------------------------------//

std::size_t SPRNG::d_packed_size = 0;

//---------------------------------------------------------------------------//
// Members.
//---------------------------------------------------------------------------//
/*!
 * \brief Deserializer constructor. Unpack a SPRNG state from a buffer.
 */
SPRNG::SPRNG( const Teuchos::ArrayView<char>& state_buffer )
    : d_stream_id( 0 )
    , d_stream( 0 )
{
    MCLS_REQUIRE( Teuchos::as<std::size_t>(state_buffer.size()) >= 
	     Teuchos::as<std::size_t>(2 * sizeof(int)) );

    Deserializer ds;
    ds.setBuffer( state_buffer );

    int rng_size = 0;
    ds >> d_stream >> rng_size;
    MCLS_CHECK( d_stream >= 0 );
    MCLS_CHECK( rng_size >= 0 );

    char* prng = new char[rng_size];
    for ( int i = 0; i < rng_size; ++i )
    {
	ds >> prng[i];
    }

    int* rng_id = unpack_sprng( prng );

    d_stream_id = new SPRNGValue( rng_id );

    delete [] prng;
    MCLS_ENSURE( ds.getPtr() == state_buffer.getRawPtr() + state_buffer.size() );
    MCLS_ENSURE( d_stream_id );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Assignment operator.
 */
SPRNG& SPRNG::operator=(const SPRNG &rhs)
{
    if ( d_stream_id == rhs.d_stream_id && d_stream == rhs.d_stream )
        return *this;

    if ( d_stream_id && --d_stream_id->d_refcount == 0 ) delete d_stream_id;

    d_stream_id = rhs.d_stream_id;
    d_stream   = rhs.d_stream;

    if ( d_stream_id ) ++d_stream_id->d_refcount;

    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Pack the SPRNG state into a buffer.
 */
Teuchos::Array<char> SPRNG::pack() const
{
    MCLS_REQUIRE( d_stream_id );

    Serializer s;
    char* prng = 0;
    int rng_size = pack_sprng( d_stream_id->d_id, &prng );
    int size = rng_size + 2 * sizeof(int);
    MCLS_CHECK( prng );

    Teuchos::Array<char> state_buffer( size );
    s.setBuffer( state_buffer() );

    s << d_stream << rng_size;

    for ( int i = 0; i < rng_size; ++i )
    {
	s << prng[i];
    }

    std::free( prng );

    MCLS_ENSURE( s.getPtr() == state_buffer.getRawPtr() + size );
    return state_buffer;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the packed size.
 */
std::size_t SPRNG::getSize() const
{
    MCLS_REQUIRE( d_stream_id );

    if ( d_packed_size > 0 )
    {
	return d_packed_size;
    }

    char *prng = 0;
    int rng_size = pack_sprng( d_stream_id->d_id, &prng );
    d_packed_size = rng_size + 2 * sizeof(int);

    std::free( prng );

    return d_packed_size;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_SPRNG.cpp
//---------------------------------------------------------------------------//

