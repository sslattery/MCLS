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
 * \file MCLS_SourceTraits.hpp
 * \author Stuart R. Slattery
 * \brief Source traits definition.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SOURCETRAITS_HPP
#define MCLS_SOURCETRAITS_HPP

#include "MCLS_RNGControl.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Array.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class UndefinedSourceTraits
 * \brief Class for undefined source traits. 
 *
 * Will throw a compile-time error if these traits are not specialized.
 */
template<class Source>
struct UndefinedSourceTraits
{
    static inline void notDefined()
    {
	return Source::this_type_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class SourceTraits
 * \brief Traits for Monte Carlo transport sources.
 *
 * SourceTraits defines an interface for parallel distributed sources.
 */
template<class Source>
class SourceTraits
{
  public:

    //@{
    //! Typedefs.
    typedef Source                                      source_type;
    typedef typename Source::ordinal_type               ordinal_type;
    typedef typename Source::history_type               history_type;
    typedef typename Source::domain_type                domain_type;
    typedef Teuchos::Comm<int>                          Comm;
    //@}

    /*!
     * \brief Create a reference-counted pointer to a new source defined over
     * the given communicator and domain by unpacking a data buffer.
     */
    static Teuchos::RCP<Source> 
    createFromBuffer( const Teuchos::ArrayView<char>& buffer,
		      const Teuchos::RCP<const Comm>& comm,
		      const Teuchos::RCP<domain_type>& domain,
		      const Teuchos::RCP<RNGControl>& rng_control,
		      const int global_comm_size,
		      const int global_comm_rank )

    { 
	UndefinedSourceTraits<Source>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Pack a source into a buffer.
     */
    static Teuchos::Array<char> pack( const Source& source )
    { 
	UndefinedSourceTraits<Source>::notDefined(); 
	return Teuchos::Array<char>(0);
    }

    /*!
     * \brief Get the size of source in packed bytes.
     */
    static std::size_t getPackedBytes( const Source& source )
    { 
	UndefinedSourceTraits<Source>::notDefined(); 
	return 0;
    }

    /*!
     * \brief Build the source.
     */
    static void buildSource( Source& source )
    {
	UndefinedSourceTraits<Source>::notDefined(); 
    }

    /*!
     * \brief Get the weight of a given on-process global state in the
     * source. 
     */
    static double weight( const Source& source, const ordinal_type state )
    { 
	UndefinedSourceTraits<Source>::notDefined(); 
	return 0.0;
    }

    /*!
     * \brief Get a history from the source.
     */
    static Teuchos::RCP<history_type> getHistory( Source& source )
    { 
	UndefinedSourceTraits<Source>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Return whether or not a source has emitted all of its
     * histories. 
     */
    static bool empty( const Source& source )
    { 
	UndefinedSourceTraits<Source>::notDefined(); 
	return false;
    }

    /*!
     * \brief Get the local number of histories to be transported by this
     * source. 
     */
    static int numToTransport( const Source& source )
    { 
	UndefinedSourceTraits<Source>::notDefined(); 
	return 0;
    }

    /*!
     * \brief Get the number of histories to be transported by this source for
     * the entire set.
     */
    static int numToTransportInSet( const Source& source )
    { 
	UndefinedSourceTraits<Source>::notDefined(); 
	return 0;
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_SOURCETRAITS_HPP

//---------------------------------------------------------------------------//
// end MCLS_SourceTraits.hpp
//---------------------------------------------------------------------------//
