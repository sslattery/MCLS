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
 * \file MCLS_AdjointTally.hpp
 * \author Stuart R. Slattery
 * \brief AdjointTally declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ADJOINTTALLY_HPP
#define MCLS_ADJOINTTALLY_HPP

#include "MCLS_DBC.hpp"
#include "MCLS_History.hpp"
#include "MCLS_VectorExport.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_TallyTraits.hpp"
#include "MCLS_Estimators.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>

#include <boost/tr1/unordered_map.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class AdjointTally
 * \brief Monte Carlo tally for the linear system solution vector for adjoint
 * problems. 
 */
template<class Vector>
class AdjointTally
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                              vector_type;
    typedef VectorTraits<Vector>                                VT;
    typedef typename VT::global_ordinal_type                    Ordinal;
    typedef typename VT::scalar_type                            Scalar;
    typedef History<Ordinal>                                    HistoryType;
    typedef Teuchos::Comm<int>                                  Comm;
    typedef typename std::tr1::unordered_map<Ordinal,int>       MapType;
    //@}

    // Constructor.
    AdjointTally( const Teuchos::RCP<Vector>& x, 
		  const Teuchos::RCP<Vector>& x_overlap,
                  const int estimator = 0 );

    // Destructor.
    ~AdjointTally()
    { /* ... */ }

    // Add a history's contribution to the tally.
    inline void tallyHistory( const HistoryType& history );

    // Combine the overlap tally with the base decomposition tally in the set.
    void combineSetTallies();

    // Combine the secondary tallies with the primary tally across a
    // block. Normalize the result with the number of sets.
    void combineBlockTallies( const Teuchos::RCP<const Comm>& block_comm,
                              const int num_sets );

    // Normalize base decomposition tally with the number of specified
    // histories.
    void normalize( const int& nh );

    // Set the base tally vector.
    void setBaseVector( const Teuchos::RCP<Vector>& x_base );

    // Zero out the tallies.
    void zeroOut();

    // Get the number global rows in the base decomposition.
    Ordinal numBaseRows() const;

    // Get the number global rows in the overlap decomposition.
    Ordinal numOverlapRows() const;

    // Get the global tally rows in the base decomposition.
    Teuchos::Array<Ordinal> baseRows() const;

    // Get the global tally rows in the overlap decomposition.
    Teuchos::Array<Ordinal> overlapRows() const;

    // Set the iteration matrix with the tally.
    void setIterationMatrix( 
        const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& h,
        const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Ordinal> >& columns,
        const Teuchos::RCP<MapType>& row_indexer );

  private:

    // Collision estimator tally.
    inline void collisionEstimatorTally( const HistoryType& history );

    // Expected value estimator tally.
    inline void expectedValueEstimatorTally( const HistoryType& history );

  private:

    // Solution vector in operator decomposition.
    Teuchos::RCP<Vector> d_x;

    // Solution vector in overlap decomposition.
    Teuchos::RCP<Vector> d_x_overlap;

    // Monte Carlo estimator type.
    int d_estimator;

    // Overlap to base decomposition vector export.
    VectorExport<Vector> d_export;

    // Local (transposed) iteration matrix values.
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > d_h;

    // Local iteration matrix global columns.
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Ordinal> > d_columns;

    // Iteration matrix local row indexer.
    Teuchos::RCP<MapType> d_row_indexer;
};

//---------------------------------------------------------------------------//
// Inline functions.
//---------------------------------------------------------------------------//
/*
 * \brief Add a history's contribution to the tally.
 */
template<class Vector>
inline void AdjointTally<Vector>::tallyHistory( const HistoryType& history )
{
    MCLS_REQUIRE( history.alive() );
    MCLS_REQUIRE( VT::isGlobalRow( *d_x, history.state() ) ||
                  VT::isGlobalRow( *d_x_overlap, history.state() ) );

    if ( COLLISION == d_estimator )
    {
        collisionEstimatorTally( history );
    }

    if ( EXPECTED_VALUE == d_estimator )
    {
        expectedValueEstimatorTally( history );
    }

    else
    {
	MCLS_INSIST( COLLISION == d_estimator || EXPECTED_VALUE == d_estimator,
                     "Estimator type not supported!" );
    }
}

//---------------------------------------------------------------------------//
/*
 * \brief Collision estimator tally.
 */
template<class Vector>
inline void AdjointTally<Vector>::collisionEstimatorTally( 
    const HistoryType& history )
{
    // The collision estimator tally sums the history's current weight into
    // x(i) where the index i is the history's current state.
    if ( VT::isGlobalRow(*d_x, history.state()) )
    {
	VT::sumIntoGlobalValue( *d_x, history.state(), history.weight() );
    }

    else if ( VT::isGlobalRow(*d_x_overlap, history.state()) )
    {
	VT::sumIntoGlobalValue( 
	    *d_x_overlap, history.state(), history.weight() );
    }

    else
    {
	MCLS_INSIST( VT::isGlobalRow(*d_x, history.state()) ||
                     VT::isGlobalRow(*d_x_overlap, history.state()),
                     "History state is not local to tally!" );
    }
}

//---------------------------------------------------------------------------//
/*
 * \brief Expected value estimator tally.
 */
template<class Vector>
inline void AdjointTally<Vector>::expectedValueEstimatorTally( 
    const HistoryType& history )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_h) );
    MCLS_REQUIRE( Teuchos::nonnull(d_columns) );
    MCLS_REQUIRE( Teuchos::nonnull(d_row_indexer) );

    // Get the local row.
    typename MapType::const_iterator index = 
	d_row_indexer->find( history.state() );
    MCLS_CHECK( index != d_row_indexer->end() );

    // The expected value estimator sums the history's weight multiplied by
    // the transpose iteration matrix value h^T_(i,j) into each tally state
    // x(j) where the index i is the history's current state.
    typename Teuchos::ArrayRCP<Ordinal>::const_iterator col_it;
    Teuchos::ArrayRCP<double>::const_iterator val_it;
    for ( col_it = d_columns[index->second].begin(),
          val_it = d_h[index->second].begin();
          col_it != d_columns[index->second].end();
          ++col_it, ++val_it )
    {
        if ( VT::isGlobalRow(*d_x, *col_it) )
        {
            VT::sumIntoGlobalValue( *d_x, *col_it, history.weight()*(*val_it) );
        }

        else if ( VT::isGlobalRow(*d_x_overlap, *col_it) )
        {
            VT::sumIntoGlobalValue( 
                *d_x_overlap, *col_it, history.weight()*(*val_it) );
        }

        else
        {
            MCLS_INSIST( VT::isGlobalRow(*d_x, *col_it) ||
                         VT::isGlobalRow(*d_x_overlap, *col_it),
                         "History state is not local to tally!" );
        }
    }
}

//---------------------------------------------------------------------------//
// TallyTraits implementation.
//---------------------------------------------------------------------------//
/*!
 * \class TallyTraits
 * \brief Specialization for AdjointTally.
 */
template<class Vector>
class TallyTraits<AdjointTally<Vector> >
{
  public:

    //@{
    //! Typedefs.
    typedef AdjointTally<Vector>                       tally_type;
    typedef typename tally_type::vector_type           vector_type;
    typedef typename tally_type::Ordinal               ordinal_type;
    typedef typename tally_type::HistoryType           history_type;
    typedef Teuchos::Comm<int>                         Comm;
    //@}

    /*!
     * \brief Add a history's contribution to the tally.
     */
    static inline void tallyHistory( tally_type& tally, 
				     const history_type& history )
    { 
	tally.tallyHistory( history );
    }

    /*!
     * \brief Combine the tallies together over a set. This is generally
     * combining the overlap and base tallies.
     */
    static void combineSetTallies( tally_type& tally )
    {
	tally.combineSetTallies();
    }

    /*!
     * \brief Combine the tallies together over a block communicator.
     */
    static void combineBlockTallies( 
	tally_type& tally,
	const Teuchos::RCP<const Comm>& block_comm,
        const int num_sets )
    {
	tally.combineBlockTallies( block_comm, num_sets );
    }

    /*!
     * \brief Normalize the tally with a specified number of histories.
     */
    static void normalize( tally_type& tally, const int nh )
    {
	tally.normalize( nh );
    }

    /*!
     * \brief Set the tally base vector. The maps are required to be
     * compatible. 
     */
    static void setBaseVector( tally_type& tally, 
			       const Teuchos::RCP<vector_type>& x_base )
    {
	tally.setBaseVector( x_base );
    }

    /*!
     * \brief Set the tallies to zero.
     */
    static void zeroOut( tally_type& tally )
    {
	tally.zeroOut();
    }

    /*!
     * \brief Get the number of global rows in the base decompostion.
     */
    static ordinal_type numBaseRows( const tally_type& tally )
    {
	return tally.numBaseRows();
    }

    /*!
     * \brief Get the number of global rows in the overlap decompostion.
     */
    static ordinal_type numOverlapRows( const tally_type& tally )
    {
	return tally.numOverlapRows();
    }

    /*!
     * \brief Get the global tally rows in the base decompostion.
     */
    static Teuchos::Array<ordinal_type> baseRows( const tally_type& tally )
    {
	return tally.baseRows();
    }

    /*!
     * \brief Get the global tally rows in the overlap decompostion.
     */
    static Teuchos::Array<ordinal_type> 
    overlapRows( const tally_type& tally )
    {
	return tally.overlapRows();
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_AdjointTally_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_ADJOINTTALLY_HPP

//---------------------------------------------------------------------------//
// end MCLS_AdjointTally.hpp
// ---------------------------------------------------------------------------//

