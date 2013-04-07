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
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_CommHelpers.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector>
AdjointTally<Vector>::AdjointTally( const Teuchos::RCP<Vector>& x, 
				    const Teuchos::RCP<Vector>& x_overlap,
                                    const int estimator )
    : d_x( x )
    , d_x_overlap( x_overlap )
    , d_estimator( estimator )
    , d_export( d_x_overlap, d_x )
{ 
    MCLS_ENSURE( !d_x.is_null() );
    MCLS_ENSURE( !d_x_overlap.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Combine the overlap tally with the operator decomposition tally in
 * the set.
 */
template<class Vector>
void AdjointTally<Vector>::combineSetTallies()
{
    d_export.doExportAdd();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Combine the base tallies across a block and normalize by the number
 * of sets.
 */
template<class Vector>
void AdjointTally<Vector>::combineBlockTallies(
    const Teuchos::RCP<const Comm>& block_comm, const int num_sets )
{
    MCLS_REQUIRE( !block_comm.is_null() );

    Teuchos::ArrayRCP<const Scalar> const_tally_view = VT::view( *d_x );

    Teuchos::ArrayRCP<Scalar> copy_buffer( const_tally_view.size() );

    Teuchos::reduceAll<int,Scalar>( *block_comm,
				    Teuchos::REDUCE_SUM,
				    Teuchos::as<int>( const_tally_view.size() ),
				    const_tally_view.getRawPtr(),
				    copy_buffer.getRawPtr() );

    Teuchos::ArrayRCP<Scalar> tally_view = VT::viewNonConst( *d_x );
    
    std::copy( copy_buffer.begin(), copy_buffer.end(), tally_view.begin() );
    VT::scale( *d_x, 1.0 / Teuchos::as<double>(num_sets) );
}

//---------------------------------------------------------------------------//
/*
 * \brief Normalize base decomposition tally with the number of specified
 * histories.
 */
template<class Vector>
void AdjointTally<Vector>::normalize( const int& nh )
{
    VT::scale( *d_x, 1.0 / Teuchos::as<double>(nh) );
}

//---------------------------------------------------------------------------//
/*
 * \brief Set the base tally vector.
 */
template<class Vector>
void AdjointTally<Vector>::setBaseVector( const Teuchos::RCP<Vector>& x_base )
{
    MCLS_REQUIRE( Teuchos::nonnull(x_base) );
    d_x = x_base;
    d_export = VectorExport<Vector>( d_x_overlap, d_x );
}

//---------------------------------------------------------------------------//
/*
 * \brief Zero out operator decomposition and overlap decomposition tallies.
 */
template<class Vector>
void AdjointTally<Vector>::zeroOut()
{
    VT::putScalar( *d_x, Teuchos::ScalarTraits<Scalar>::zero() );
    VT::putScalar( *d_x_overlap, Teuchos::ScalarTraits<Scalar>::zero() );
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
    return VT::getLocalLength( *d_x_overlap );
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
/*!
 * \brief Set the iteration matrix with the tally
 */
template<class Vector>
void AdjointTally<Vector>::setIterationMatrix( 
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& h,
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Ordinal> >& columns,
    const Teuchos::RCP<MapType>& row_indexer )
{
    MCLS_REQUIRE( EXPECTED_VALUE == d_estimator );
    d_h = h;
    d_columns = columns;
    d_row_indexer = row_indexer;
    MCLS_ENSURE( Teuchos::nonnull(d_row_indexer) );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_ADJOINTTALLY_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_AdjointTally_impl.hpp
// ---------------------------------------------------------------------------//

