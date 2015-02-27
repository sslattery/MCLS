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
 * \file MCLS_History_impl.hpp
 * \author Stuart R. Slattery
 * \brief History class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_HISTORY_IMPL_HPP
#define MCLS_HISTORY_IMPL_HPP

#include <algorithm>

#include "MCLS_DBC.hpp"

#include <Teuchos_as.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Pack the history into a buffer.
 */
template<class Ordinal>
void History<Ordinal>::packHistory( Serializer& s ) const
{
    s << b_global_state << b_weight << static_cast<int>(b_alive)
      << b_event << b_num_steps;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Unpack the history from a buffer.
 */
template<class Ordinal>
void History<Ordinal>::unpackHistory( Deserializer& ds )
{
    int alive;
    ds >> b_global_state >> b_weight >> alive >> b_event >> b_num_steps;
    b_alive = static_cast<bool>(alive);
}

//---------------------------------------------------------------------------//
// Static members.
//---------------------------------------------------------------------------//
template<class Ordinal>
std::size_t History<Ordinal>::b_packed_bytes = 0;

//---------------------------------------------------------------------------//
/*!
 * \brief Set the byte size of the packed history state.
 */
template<class Ordinal>
void History<Ordinal>::setStaticSize()
{
    b_packed_bytes = sizeof(Ordinal) + sizeof(double) + 3*sizeof(int);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the number of bytes in the packed history state.
 */
template<class Ordinal>
std::size_t History<Ordinal>::getStaticSize()
{
    MCLS_REQUIRE( b_packed_bytes );
    return b_packed_bytes;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_HISTORY_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_History_impl.hpp
//---------------------------------------------------------------------------//

