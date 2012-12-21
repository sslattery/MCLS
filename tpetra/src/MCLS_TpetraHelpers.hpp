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
 * \file MCLS_TpetraAdapater.hpp
 * \author Stuart R. Slattery
 * \brief Tpetra Helpers.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_TPETRAHELPERS_HPP
#define MCLS_TPETRAHELPERS_HPP

#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_VbrMatrix.hpp>

//---------------------------------------------------------------------------//
/*!
 * \class UndefinedTpetraHelpers
 * \brief Class for undefined TpetraHelper functions.
 *
 * Will throw a compile-time error if these traits are not specialized.
 */
template<class OperatorType>
struct UndefinedTpetraHelpers
{
    static inline OperatorType notDefined()
    {
	return OperatorType::this_type_is_missing_a_specialization();
    }
};

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class TpetraHelpers
 * \brief Helper functions for Tpetra implementations.
 */
template<class OperatorType>
class TpetraHelpers
{
  public:

    //@{
    //! Typedefs.
    typedef OperatorType                                  operator_type;
    typedef typename operator_type::scalar_type           scalar_type;
    typedef typename operator_type::local_ordinal_type    local_ordinal_type;
    typedef typename operator_type::global_ordinal_type   global_ordinal_type;
    //@}

    /*!
     * \brief Get the on process global matrix row indices that, as global
     * column indices, are off process.
     */
    static Teuchos::Array<global_ordinal_type> 
    getOffProcRowsAsCols( const operator_type& op )
    { 
	UndefinedTpetraHelpers<operator_type>::notDefined(); 
	return Teuchos::Array<global_ordinal_type>(0); 
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class TpetraHelpers
 * \brief TpetraHelpers specialization for Tpetra::CrsMatrix.
 */
template<class Scalar, class LO, class GO>
class TpetraHelpers<Tpetra::CrsMatrix<Scalar,LO,GO> >
{
  public:

    //@{
    //! Typedefs.
    typedef Tpetra::CrsMatrix<Scalar,LO,GO>               operator_type;
    typedef Scalar                                        scalar_type;
    typedef LO                                            local_ordinal_type;
    typedef GO                                            global_ordinal_type;
    //@}

    /*!
     * \brief Get the on process global matrix row indices that, as global
     * column indices, are off process.
     */
    static Teuchos::Array<global_ordinal_type> 
    getOffProcRowsAsCols( const operator_type& op )
    { 
	Teuchos::RCP<const Tpetra::Map<LO,GO> > row_map = op.getRowMap();
	Teuchos::RCP<const Tpetra::Map<LO,GO> > col_map = op.getColMap();

	Teuchos::ArrayView<const GO> global_cols = 
	    col_map->getNodeElementList();

	Teuchos::Array<GO> off_proc_cols(0);
	typename Teuchos::ArrayView<const GO>::const_iterator global_col_it;
	for ( global_col_it = global_cols.begin();
	      global_col_it != global_cols.end();
	      ++global_col_it )
	{
	    if ( !row_map->isNodeGlobalElement( *global_col_it ) )
	    {
		off_proc_cols.push_back( *global_col_it );
	    }
	}

	return off_proc_cols;
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class TpetraHelpers
 * \brief TpetraHelpers specialization for Tpetra::VbrMatrix.
 */
template<class Scalar, class LO, class GO>
class TpetraHelpers<Tpetra::VbrMatrix<Scalar,LO,GO> >
{
  public:

    //@{
    //! Typedefs.
    typedef Tpetra::VbrMatrix<Scalar,LO,GO>               operator_type;
    typedef Scalar                                        scalar_type;
    typedef LO                                            local_ordinal_type;
    typedef GO                                            global_ordinal_type;
    //@}

    /*!
     * \brief Get the on process global matrix row indices that, as global
     * column indices, are off process.
     */
    static Teuchos::Array<global_ordinal_type> 
    getOffProcRowsAsCols( const operator_type& op )
    { 
	Teuchos::RCP<const Tpetra::Map<LO,GO> > row_map = op.getPointRowMap();
	Teuchos::RCP<const Tpetra::Map<LO,GO> > col_map = op.getPointColMap();

	Teuchos::ArrayView<const GO> global_cols = 
	    col_map->getNodeElementList();

	Teuchos::Array<GO> off_proc_cols(0);
	typename Teuchos::ArrayView<const GO>::const_iterator global_col_it;
	for ( global_col_it = global_cols.begin();
	      global_col_it != global_cols.end();
	      ++global_col_it )
	{
	    if ( !row_map->isNodeGlobalElement( *global_col_it ) )
	    {
		off_proc_cols.push_back( *global_col_it );
	    }
	}

	return off_proc_cols;
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_TPETRAHELPERS_HPP

//---------------------------------------------------------------------------//
// end MCLS_TpetraAdapater.hpp
//---------------------------------------------------------------------------//
