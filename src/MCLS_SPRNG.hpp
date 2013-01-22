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
 * \file MCLS_SPRNG.hpp
 * \author Stuart R. Slattery
 * \brief SPRNG wrapper class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SPRNG_HPP
#define MCLS_SPRNG_HPP

#include "MCLS_DBC.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Array.hpp>

#include <sprng.h>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class SPRNG
 * \brief A wrapper class for managing the SPRNG library. This class was
 * developed by Tom Evans. 
 */
//---------------------------------------------------------------------------//
class SPRNG
{
  private:

    //! Reference counting container for SPRNG memory.
    struct SPRNGValue
    {
        // SPRNG library id.
        int *d_id;

	// Reference counter.
	int d_refcount;

        // Constructor.
        SPRNGValue( int *id ) 
	    : d_id( id )
	    , d_refcount( 1 )
	{ /* ... */ }

        // Destructor.
        ~SPRNGValue()
	{ free_sprng( d_id ); }
    };

  public:
    
    //! Default constructor.
    inline SPRNG()
	: d_stream_id( 0 )
	, d_stream( 0 )
    { /* ... */ }

    //! State constructor.
    inline SPRNG( int *id_val, int number )
	: d_stream_id( new SPRNGValue(id_val) )
	, d_stream( number )
    { /* ... */ }

    // Deserializer constructor.
    SPRNG( const Teuchos::ArrayView<char>& state_buffer );

    //! Copy constructor.
    inline SPRNG( const SPRNG& rhs )
	: d_stream_id( rhs.d_stream_id )
	, d_stream( rhs.d_stream )
    { if ( d_stream_id ) ++ d_stream_id->d_refcount; }
    
    //! Desctructor.
    inline ~SPRNG()
    {
	if ( d_stream_id && --d_stream_id->d_refcount == 0 )
	    delete d_stream_id;
    }

    // Assignment operator.
    SPRNG& operator=(const SPRNG &);

    //! Check if this SPRNG object has been assigned a stream.
    bool assigned() const
    { return (d_stream_id != 0); }

    // Pack the SPRNG state into a buffer.
    Teuchos::Array<char> pack() const;

    //! Get a random number.
    double random() const
    {
	Require( d_stream_id );
	return sprng( d_stream_id->d_id );
    }

    //! Get the SPRNG ID pointer.
    int* getID() const
    {
	Require( d_stream_id );
	return d_stream_id->d_id;
    }

    //! Get the stream number index.
    int getIndex() const
    {
	Require( d_stream_id );
	return d_stream;
    }

    // Get the packed size.
    std::size_t getSize() const;

    //! Print diagnostics.
    void print() const
    {
	Require( d_stream_id );
	print_sprng( d_stream_id->d_id );
    }

  private:

    // SPRNG library memory.
    SPRNGValue* d_stream_id;

    // Stream number.
    int d_stream;

    // Size of SPRNG data in packed state.
    static std::size_t d_packed_size;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_SPRNG_HPP

//---------------------------------------------------------------------------//
// end MCLS_SPRNG.hpp
//---------------------------------------------------------------------------//

