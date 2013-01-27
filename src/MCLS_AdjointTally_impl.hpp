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
 * \file MCLS_AdjointTally_impl.hpp
 * \author Stuart R. Slattery
 * \brief AdjointTally implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ADJOINTTALLY_IMPL_HPP
#define MCLS_ADJOINTTALLY_IMPL_HPP

#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_OrdinalTraits.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector>
AdjointTally<Vector>::AdjointTally( const Teuchos::RCP<Vector>& x, 
				    const Teuchos::RCP<Vector>& x_overlap )
    : d_x( x )
    , d_x_overlap( x_overlap )
    , d_export( d_x_overlap, d_x )
{ 
    Ensure( !d_x.is_null() );
    Ensure( !d_x_overlap.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Combine the overlap tally with the base decomposition tally in the
 * set. 
 */
template<class Vector>
void AdjointTally<Vector>::combineSetTallies()
{
    d_export.doExportAdd();
}

//---------------------------------------------------------------------------//
/*
 * \brief Normalize base decomposition tally with the number of specified
 * histories.
 */
template<class Vector>
void AdjointTally<Vector>::normalize( const int& nh )
{
    VT::scale( *d_x, 1.0 / nh );
}

//---------------------------------------------------------------------------//
/*
 * \brief Zero out base decomposition and overlap decomposition tallies.
 */
template<class Vector>
void AdjointTally<Vector>::zeroOut()
{
    VT::putScalar( *d_x, 
		   Teuchos::ScalarTraits<typename VT::scalar_type>::zero() );

    VT::putScalar( *d_x_overlap, 
		   Teuchos::ScalarTraits<typename VT::scalar_type>::zero() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the number global rows in the base decomposition.
 */
template<class Vector>
typename AdjointTally<Vector>::Ordinal 
AdjointTally<Vector>::numBaseRows() const
{
    return VT::getLocalLength( *d_x );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the number global rows in the overlap decomposition.
 */
template<class Vector>
typename AdjointTally<Vector>::Ordinal 
AdjointTally<Vector>::numOverlapRows() const
{
    return VT::getLocalLength( *d_x );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the global rows in the base decomposition.
 */
template<class Vector>
Teuchos::Array<typename AdjointTally<Vector>::Ordinal>
AdjointTally<Vector>::baseRows() const
{
    Teuchos::Array<Ordinal> base_rows( VT::getLocalLength(*d_x) );
    typename Teuchos::Array<Ordinal>::iterator row_it;
    typename VT::local_ordinal_type local_row = 
	Teuchos::OrdinalTraits<typename VT::local_ordinal_type>::zero();
    for ( row_it = base_rows.begin();
	  row_it != base_rows.end();
	  ++row_it )
    {
	*row_it = VT::getGlobalRow( *d_x, local_row );
	++local_row;
    }

    return base_rows;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the global rows in the overlap decomposition.
 */
template<class Vector>
Teuchos::Array<typename AdjointTally<Vector>::Ordinal>
AdjointTally<Vector>::overlapRows() const
{
    Teuchos::Array<Ordinal> overlap_rows( VT::getLocalLength(*d_x_overlap) );
    typename Teuchos::Array<Ordinal>::iterator row_it;
    typename VT::local_ordinal_type local_row = 
	Teuchos::OrdinalTraits<typename VT::local_ordinal_type>::zero();
    for ( row_it = overlap_rows.begin();
	  row_it != overlap_rows.end();
	  ++row_it )
    {
	*row_it = VT::getGlobalRow( *d_x_overlap, local_row );
	++local_row;
    }

    return overlap_rows;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_ADJOINTTALLY_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_AdjointTally_impl.hpp
// ---------------------------------------------------------------------------//

