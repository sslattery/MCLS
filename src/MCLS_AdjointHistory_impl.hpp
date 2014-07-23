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
 * \file MCLS_AdjointHistory_impl.hpp
 * \author Stuart R. Slattery
 * \brief AdjointHistory class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ADJOINTHISTORY_IMPL_HPP
#define MCLS_ADJOINTHISTORY_IMPL_HPP

#include <algorithm>

#include "MCLS_DBC.hpp"
#include "MCLS_Serializer.hpp"

#include <Teuchos_as.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Deserializer constructor.
 */
template<class Ordinal>
AdjointHistory<Ordinal>::AdjointHistory( const Teuchos::ArrayView<char>& buffer )
{
    MCLS_REQUIRE( Teuchos::as<std::size_t>(buffer.size()) == d_packed_bytes );

    // If we sent the RNG state with the history, unpack that first.
    if ( d_packed_rng > 0 )
    {
	Teuchos::Array<char> brng( d_packed_rng );
	std::copy( &buffer[0], &buffer[d_packed_rng], brng.begin() );
	d_rng = RNG( brng );
    }

    // Unpack the state of the history.
    Deserializer ds;
    ds.setBuffer( d_packed_bytes - d_packed_rng, &buffer[d_packed_rng] );
    int balive;
    ds >> d_state >> d_weight >> balive >> d_event;
    d_alive = static_cast<bool>(balive);

    MCLS_ENSURE( ds.getPtr() == ds.end() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Pack the history into a buffer.
 */
template<class Ordinal>
Teuchos::Array<char> AdjointHistory<Ordinal>::pack() const
{
    MCLS_REQUIRE( d_packed_bytes );
    MCLS_REQUIRE( d_packed_bytes - d_packed_rng > 0 );

    Teuchos::Array<char> buffer( d_packed_bytes );

    if ( d_packed_rng > 0 )
    {
	Teuchos::Array<char> brng = d_rng.pack();
	MCLS_CHECK( Teuchos::as<std::size_t>(brng.size()) == d_packed_rng );

	std::copy( brng.begin(), brng.end(), buffer.begin() );
    }

    Serializer s;
    s.setBuffer( d_packed_bytes - d_packed_rng, &buffer[d_packed_rng] );
    s << d_state << d_weight << static_cast<int>(d_alive) << d_event;

    MCLS_ENSURE( s.getPtr() == s.end() );
    return buffer;
}

//---------------------------------------------------------------------------//
// Static members.
//---------------------------------------------------------------------------//
template<class Ordinal>
std::size_t AdjointHistory<Ordinal>::d_packed_bytes = 0;

template<class Ordinal>
std::size_t AdjointHistory<Ordinal>::d_packed_rng = 0;

//---------------------------------------------------------------------------//
/*!
 * \brief Set the byte size of the packed history state.
 */
template<class Ordinal>
void AdjointHistory<Ordinal>::setByteSize( std::size_t size_rng_state )
{
    d_packed_rng = size_rng_state;
    d_packed_bytes = d_packed_rng + sizeof(Ordinal) + sizeof(double)
		     + 2*sizeof(int);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the number of bytes in the packed history state.
 */
template<class Ordinal>
std::size_t AdjointHistory<Ordinal>::getPackedBytes()
{
    MCLS_REQUIRE( d_packed_bytes );
    return d_packed_bytes;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_ADJOINTHISTORY_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_AdjointHistory_impl.hpp
//---------------------------------------------------------------------------//

