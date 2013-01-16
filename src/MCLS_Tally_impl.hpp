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
 * \file MCLS_Tally_impl.hpp
 * \author Stuart R. Slattery
 * \brief Tally implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_TALLY_IMPL_HPP
#define MCLS_TALLY_IMPL_HPP

#include <MCLS_DBC.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector>
Tally<Vector>::Tally( const Teuchos::RCP<Vector>& x, 
		      const Teuchos::RCP<Vector>& x_overlap )
    : d_x( x )
    , d_x_overlap( x_overlap )
    , d_export( d_x_overlap, d_x )
{ 
    Ensure( !d_x.is_null() );
    Ensure( !d_x_overlap.is_null() );
}

//---------------------------------------------------------------------------//
/*
 * \brief Add a history's contribution to the tally.
 */
template<class Vector>
void Tally<Vector>::tallyHistory( const HistoryType& history )
{
    Require( VT::isGlobalRow( *d_x, history.state() ) ||
	     VT::isGlobalRow( *d_x_overlap, history.state() ) );

    if ( VT::isGlobalRow( *d_x, history.state() ) )
    {
	VT::sumIntoGlobalValue( *d_x, history.state(), history.weight() );
    }

    else if ( VT::isGlobalRow( *d_x_overlap, history.state() ) )
    {
	VT::sumIntoGlobalValue( 
	    *d_x_overlap, history.state(), history.weight() );
    }

    else
    {
	Insist( VT::isGlobalRow( *d_x, history.state() ) ||
		VT::isGlobalRow( *d_x_overlap, history.state() ),
		"History state is not local to tally!" );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Combine the overlap tally with the base decomposition tally.
 */
template<class Vector>
void Tally<Vector>::combineTallies()
{
    d_export.doExportAdd();
}

//---------------------------------------------------------------------------//
/*
 * \brief Normalize base decomposition tallies with the number of specified
 * histories.
 */
template<class Vector>
void Tally<Vector>::normalize( const int& nh )
{
    VT::scale( *d_x, 1.0 / nh )
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_TALLY_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_Tally_impl.hpp
// ---------------------------------------------------------------------------//

