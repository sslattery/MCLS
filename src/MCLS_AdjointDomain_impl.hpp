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
 * \file MCLS_AdjointDomain_impl.hpp
 * \author Stuart R. Slattery
 * \brief AdjointDomain implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ADJOINTDOMAIN_IMPL_HPP
#define MCLS_ADJOINTDOMAIN_IMPL_HPP

#include <algorithm>
#include <limits>
#include <string>

#include "MCLS_Serializer.hpp"
#include "MCLS_Estimators.hpp"
#include "MCLS_VectorExport.hpp"

#include <Teuchos_as.hpp>
#include <Teuchos_Array.hpp>

#include <Tpetra_Distributor.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Matrix constructor.
 */
template<class Vector, class Matrix, class RNG>
AdjointDomain<Vector,Matrix,RNG>::AdjointDomain( 
    const Teuchos::RCP<const Matrix>& A,
    const Teuchos::RCP<Vector>& x,
    const Teuchos::ParameterList& plist )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );
    MCLS_REQUIRE( Teuchos::nonnull(x) );

    // Build the domain data.
    Teuchos::Array<Ordinal> local_tally_states;
    this->buildDomain( A, x, plist, local_tally_states );

    // Create the tally vector.
    Teuchos::RCP<Vector> x_tally =
        VT::createFromRows( MT::getComm(*A), local_tally_states() );

    // Create the tally.
    this->b_tally = Teuchos::rcp( new TallyType(x, x_tally, this->b_estimator) );

    // If we are using the expected value estimator, provide the iteration
    // matrix to the tally. 
    if ( Estimator::EXPECTED_VALUE == this->b_estimator )
    {
        this->b_tally->setIterationMatrix( this->b_h, this->b_local_columns );
    }

    MCLS_ENSURE( Teuchos::nonnull(this->b_tally) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Deserializer constructor.
 * 
 * \param buffer Data buffer to construct the domain from.
 *
 * \param set_comm Set constant communicator for this domain over which to
 * reconstruct the tallies.
 */
template<class Vector, class Matrix, class RNG>
AdjointDomain<Vector,Matrix,RNG>::AdjointDomain( 
    const Teuchos::ArrayView<char>& buffer,
    const Teuchos::RCP<const Comm>& set_comm )
{
    Deserializer ds;
    ds.setBuffer( buffer() );

    // Unpack the domain data.
    Teuchos::Array<Ordinal> base_rows;
    this->unpackDomain( ds, base_rows );

    // Unpack the number of tally rows in the tally.
    Ordinal num_tally = 0;
    ds >> num_tally;
    MCLS_CHECK( num_tally > 0 );

    // Unpack the tally tally rows.
    Teuchos::Array<Ordinal> tally_rows( num_tally );
    typename Teuchos::Array<Ordinal>::iterator tally_it;
    for ( tally_it = tally_rows.begin();
	  tally_it != tally_rows.end();
	  ++tally_it )
    {
	ds >> *tally_it;
    }
    MCLS_CHECK( ds.end() == ds.getPtr() );

    // Create the tally.
    Teuchos::RCP<Vector> base_x = VT::createFromRows( set_comm, base_rows() );
    Teuchos::RCP<Vector> tally_x = VT::createFromRows( set_comm, tally_rows() );
    this->b_tally = Teuchos::rcp( 
	new TallyType(base_x, tally_x, this->b_estimator) );

    // Set the iteration matrix data with the tally if using the expected
    // value estimator.
    if ( Estimator::EXPECTED_VALUE == this->b_estimator )
    {
        this->b_tally->setIterationMatrix( this->b_h, this->b_local_columns );
    }

    MCLS_ENSURE( Teuchos::nonnull(this->b_tally) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Pack the domain into a buffer.
 */
template<class Vector, class Matrix, class RNG>
Teuchos::Array<char> AdjointDomain<Vector,Matrix,RNG>::pack() const
{
    // Get the byte size of the buffer.
    std::size_t packed_bytes = getPackedBytes();
    MCLS_CHECK( packed_bytes );

    // Build the buffer and set it with the serializer.
    Teuchos::Array<char> buffer( packed_bytes );
    Serializer s;
    s.setBuffer( buffer() );

    // Pack the domain data.
    this->packDomain( s );

    // Pack the number of tally rows in the tally.
    s << Teuchos::as<Ordinal>(this->b_tally->numTallyRows());

    // Pack up the tally tally rows.
    Teuchos::Array<Ordinal> tally_rows = this->b_tally->tallyRows();
    typename Teuchos::Array<Ordinal>::const_iterator tally_it;
    for ( tally_it = tally_rows.begin();
	  tally_it != tally_rows.end();
	  ++tally_it )
    {
	s << *tally_it;
    }
    MCLS_ENSURE( s.end() == s.getPtr() );

    return buffer;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the size of this object in packed bytes.
 */
template<class Vector, class Matrix, class RNG>
std::size_t AdjointDomain<Vector,Matrix,RNG>::getPackedBytes() const
{
    Serializer s;
    s.computeBufferSizeMode();

    // Pack in the domain data.
    this->packDomain( s );

    // Pack in the number of tally rows in the tally.
    s << Teuchos::as<Ordinal>(this->b_tally->numTallyRows());

    // Pack up the tally tally rows.
    Teuchos::Array<Ordinal> tally_rows = this->b_tally->tallyRows();
    typename Teuchos::Array<Ordinal>::const_iterator tally_it;
    for ( tally_it = tally_rows.begin();
	  tally_it != tally_rows.end();
	  ++tally_it )
    {
	s << *tally_it;
    }

    return s.size();
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_ADJOINTDOMAIN_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_AdjointDomain_impl.hpp
// ---------------------------------------------------------------------------//
