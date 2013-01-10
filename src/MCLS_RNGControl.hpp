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
 * \file MCLS_RNGControl.hpp
 * \author Stuart R. Slattery
 * \brief RNGControl class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_RNGCONTROL_HPP
#define MCLS_RNGCONTROL_HPP

#include "MCLS_DBC.hpp"
#include "MCLS_SPRNG.hpp"

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class RNGControl
 * \brief Manager class for the SPRNG library. Identical to that developed by
 * Tom Evans. 
 */
//---------------------------------------------------------------------------//
class RNGControl
{
  public:

    //@{
    //! Typedefs.
    typedef SPRNG RNG;
    //@}

    // Constructor.
    RNGControl( int seed, int number = 1000000000, int stream = 0,
		int parameter = 1 );

    // Destructor.
    ~RNGControl()
    { /* ... */ }

    // Create a SPRNG object.
    RNG rng();

    // Create a SPRNG object with a specified stream index.
    RNG rng( int stream );

    // Spawn a SPRNG object.
    RNG spawn( const RNG& random );

    //! Get the current random number stream index.
    int getIndex() const
    { return d_stream; }

    //! Set the current random number stream index.
    void setIndex( int stream )
    {
	Require( stream < d_number );
	d_stream = stream; 
    }

    //! Get the size of the packed random number state.
    std::size_t getSize() const 
    {
	return d_size;
    }

    //! Get the seed value for SPRNG.
    int getSeed() const
    { 
	return d_seed;
    }

    //! Return the total number of current streams set.
    int getNumber() const
    { 
	return d_number;
    }

  private:

    // Make a SPRNG object.
    inline RNG createRNG() const;

  private:

    // Seed for SPRNG stream initialization.
    int d_seed;

    // Total number of streams.
    int d_number;
    
    // Index of current stream.
    int d_stream;

    // Control parameter for stream initialization.
    int d_parameter;

    // Size of packed stream state.
    std::size_t d_size;
};

//---------------------------------------------------------------------------//
// Inline functions.
//---------------------------------------------------------------------------//
inline RNGControl::RNG RNGControl::createRNG() const
{
    Require( d_stream <= d_number );

    int *id = init_sprng( d_stream, d_number, d_seed, d_parameter );
    RNG random( id, d_stream );

    return random;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_RNGCONTROL_HPP

//---------------------------------------------------------------------------//
// end MCLS_RNGControl.hpp
//---------------------------------------------------------------------------//

