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
 * \file MCLS_AdjointHistory.hpp
 * \author Stuart R. Slattery
 * \brief Adjoint history class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ADJOINTHISTORY_HPP
#define MCLS_ADJOINTHISTORY_HPP

#include <cmath>

#include "MCLS_PRNG.hpp"
#include "MCLS_HistoryTraits.hpp"

#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Array.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class History
 * \brief Encapsulation of a random walk history's state for adjoint
 * calculations.
 */
//---------------------------------------------------------------------------//
template<class Ordinal, class RNG>
class AdjointHistory
{
  public:

    //@{
    //! Typedefs.
    typedef Ordinal                                   ordinal_type;
    typedef RNG                                       rng_type;
    //@}

    //! Default constructor.
    AdjointHistory()
	: d_state( Teuchos::OrdinalTraits<Ordinal>::zero() )
	, d_weight( Teuchos::ScalarTraits<double>::one() )
	, d_alive( false )
	, d_event( 0 )
    { /* ... */ }

    //! State constructor.
    AdjointHistory( Ordinal state, double weight )
	: d_state( state )
	, d_weight( weight )
	, d_alive( false )
	, d_event( 0 )
    { /* ... */ }

    // Deserializer constructor.
    explicit AdjointHistory( const Teuchos::ArrayView<char>& buffer );

    // Destructor.
    ~AdjointHistory()
    { /* ... */ }

    // Pack the history into a buffer.
    Teuchos::Array<char> pack() const;

    //! Set the history state.
    inline void setState( const Ordinal state )
    { d_state = state; }

    //! Get the history state.
    inline Ordinal state() const 
    { return d_state; }

    //! Set the history weight.
    inline void setWeight( const double weight )
    { d_weight = weight; }

    //! Add to the history weight.
    inline void addWeight( const double weight )
    { d_weight += weight; }

    //! Multiply the history weight.
    inline void multiplyWeight( const double weight )
    { d_weight *= weight; }

    //! Get the history weight.
    inline double weight() const
    { return d_weight; }

    //! Get the absolute value of the history weight.
    inline double weightAbs() const
    { return std::abs(d_weight); }

    //! Set a new random number generator.
    void setRNG( const Teuchos::RCP<PRNG<RNG> >& rng )
    { d_rng = rng; }

    //! Get the random number generator.
    Teuchos::RCP<PRNG<RNG> > rng() const
    { return d_rng; }

    //! Kill the history.
    void kill()
    { d_alive = false; }

    //! Set the history alive.
    void live()
    { d_alive = true; }

    //! Get the history live/dead status.
    bool alive() const
    { return d_alive; }

    //! Set the event flag.
    void setEvent( const int event )
    { d_event = event; }

    //! Get the last event.
    int event() const
    { return d_event; }

  public:

    // Set the byte size of the packed history state.
    static void setByteSize();

    // Get the number of bytes in the packed history state.
    static std::size_t getPackedBytes();

  private:

    //  history state.
    Ordinal d_state;

    // AdjointHistory weight.
    double d_weight;

    // Random number generator.
    Teuchos::RCP<PRNG<RNG> > d_rng;

    // Alive/dead status.
    bool d_alive;

    // Latest history event.
    int d_event;

  private:

    // Packed size of history in bytes.
    static std::size_t d_packed_bytes;
};

//---------------------------------------------------------------------------//
// HistoryTraits Implementation.
//---------------------------------------------------------------------------//
template<class Ordinal,class RNG>
class HistoryTraits<AdjointHistory<Ordinal,RNG> >
{
  public:

    //@{
    //! Typedefs.
    typedef AdjointHistory<Ordinal,RNG>                  history_type;
    typedef typename history_type::ordinal_type          ordinal_type;
    typedef typename history_type::rng_type              rng_type;
    //@}

    /*!
     * \brief Create a history from a buffer.
     */
    static Teuchos::RCP<history_type> 
    createFromBuffer( const Teuchos::ArrayView<char>& buffer )
    { 
	return Teuchos::rcp( new history_type(buffer) );
    }

    /*!
     * \brief Pack the history into a buffer.
     */
    static Teuchos::Array<char> pack( const history_type& history )
    {
	return history.pack();
    }

    /*!
     * \brief Set the state of a history
     */
    static inline void setState( history_type& history, 
				 const ordinal_type state )
    {
	history.setState( state );
    }

    /*! 
     * \brief get the state of a history.
     */
    static inline ordinal_type state( const history_type& history )
    {
	return history.state();
    }

    /*!
     * \brief Set the history weight.
     */
    static inline void setWeight( history_type& history, const double weight )
    {
	history.setWeight( weight );
    }

    /*! 
     * \brief Add to the history weight.
     */
    static inline void addWeight( history_type& history, const double weight )
    {
	history.addWeight( weight );
    }

    /*! 
     * \brief Multiply the history weight.
     */
    static inline void multiplyWeight( history_type& history, 
				       const double weight )
    {
	history.multiplyWeight( weight );
    }

    /*!
     * \brief Get the history weight.
     */
    static inline double weight( const history_type& history )
    {
	return history.weight();
    }

    /*!
     * \brief Get the absolute value of the history weight.
     */
    static inline double weightAbs( const history_type& history )
    {
	return history.weightAbs();
    }

    /*!
     * \brief Set a new random number generator with the history.
     */
    static void setRNG( history_type& history, 
			const Teuchos::RCP<PRNG<rng_type> >& rng )
    {
	history.setRNG( rng );
    }

    /*!
     * \brief Get this history's random number generator.
     */
    static Teuchos::RCP<PRNG<rng_type> > rng( const history_type& history )
    {
	return history.rng();
    }

    /*!
     * \brief Kill the history.
     */
    static void kill( history_type& history )
    {
	history.kill();
    }

    /*!
     * \brief Set the history alive
     */
    static void live( history_type& history )
    {
	history.live();
    }

    /*!
     * \brief Get the history live/dead status.
     */
    static bool alive( const history_type& history )
    {
	return history.alive();
    }

    /*!
     * \brief Set the event flag.
     */
    static void setEvent( history_type& history, const int event )
    {
	history.setEvent( event );
    }

    /*!
     * \brief Get the last event.
     */
    static int event( const history_type& history )
    {
	return history.event();
    }

    /*!
     * \brief Set the byte size of the packed history state.
     */
    static void setByteSize()
    {
	history_type::setByteSize();
    }

    /*!
     * \brief Get the number of bytes in the packed history state.
     */
    static std::size_t getPackedBytes()
    {
	return history_type::getPackedBytes();
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_AdjointHistory_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_ADJOINTHISTORY_HPP

//---------------------------------------------------------------------------//
// end MCLS_AdjointHistory.hpp
//---------------------------------------------------------------------------//

