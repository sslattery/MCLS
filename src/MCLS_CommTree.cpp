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
 * \file MCLS_CommTree.cpp
 * \author Stuart R. Slattery
 * \brief Binary communication tree class definition.
 */
//---------------------------------------------------------------------------//

#include "MCLS_DBC.hpp"
#include "MCLS_CommTree.hpp"

#include <Teuchos_OrdinalTraits.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
CommTree::CommTree( const Teuchos::RCP<const Comm>& comm )
    : d_comm( comm)
    , d_parent( Teuchos::OrdinalTraits<int>::invalid() )
    , d_children( Teuchos::OrdinalTraits<int>::invalid(),
		  Teuchos::OrdinalTraits<int>::invalid() )
{
    // Get the comm parameters.
    int my_rank = d_comm->getRank();
    int tree_size = d_comm->getSize();

    // Get the parent. Proc 0 has no parent.
    if ( my_rank != 0 )
    {
	if ( my_rank % 2 == 0 )
	{
	    d_parent = ( my_rank / 2 ) - 1;
	}
	else
	{
	    d_parent = ( my_rank - 1 ) / 2;
	}
    }
	 
    // Get the first child.
    int child_1 = ( my_rank * 2 ) + 1;
    if ( child_1 < tree_size )
    {
	d_children.first = child_1;
    }

    // Get the second child.
    int child_2 = child_1 + 1;
    if ( child_2 < tree_size )
    {
	d_children.second = child_2;
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Destructor.
 */
CommTree::~CommTree()
{ /* ... */ }

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_CommTree.cpp
//---------------------------------------------------------------------------//

