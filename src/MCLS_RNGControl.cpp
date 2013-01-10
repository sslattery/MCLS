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
 * \file MCLS_RNGControl.cpp
 * \author Stuart R. Slattery
 * \brief RNGControl class implementation.
 */
//---------------------------------------------------------------------------//

#include "MCLS_RNGControl.hpp"

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
RNGControl::RNGControl( int seed, int number, int stream, int parameter )
    : d_seed( seed )
    , d_number( number )
    , d_stream( stream )
    , d_parameter( parameter )
{
    Require( d_stream <= d_number );

    RNG temp = createRNG();
    d_size = temp.getSize();

    Ensure( d_size >= 0 );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Create a SPRNG object.
 */
RNGControl::RNG RNGControl::rng()
{
    Require( d_stream <= d_number );

    RNG random = createRNG();
    ++d_stream;

    return random;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Create a SPRNG object with a specified stream index.
 */
RNGControl::RNG RNGControl::rng( int stream )
{
    Require( stream <= d_number );

    d_stream = stream;
    RNG random = createRNG();
    ++d_stream;

    return random;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Spawn a SPRNG object.
 */
RNGControl::RNG RNGControl::spawn( const RNG& random )
{
    int **new_stream;
    spawn_sprng( random.getID(), 1, &new_stream );
    RNG new_random( new_stream[0], random.getIndex() );

    std::free( new_stream );

    return new_random;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_RNGControl.cpp
//---------------------------------------------------------------------------//

