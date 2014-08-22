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
 * \file MCLS_MCSAStatusTest.hpp
 * \author Stuart R. Slattery
 * \brief MCSA NOX status test.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MCSASTATUSTEST_HPP
#define MCLS_MCSASTATUSTEST_HPP

#include <NOX_StatusTest_Generic.H>
#include <NOX_Abstract_Vector.H>
#include <NOX_Utils.H>

// Forward declaration
namespace NOX {
namespace Abstract {
class Group;
}
}

namespace MCLS 
{

//---------------------------------------------------------------------------//
/*!
 * \class MCSAStatusTest
 * \brief NOX status test for MCSA.
 */
template<class Vector, class Matrix>
class MCSAStatusTest : public NOX::StatusTest::Generic 
{

  public:

    MCSAStatusTest( double tolerance,
		    const NOX::Utils* u = NULL );

    // derived
    virtual NOX::StatusTest::StatusType
    checkStatus( const NOX::Solver::Generic& problem,
		 NOX::StatusTest::CheckType checkType );

    // derived
    virtual NOX::StatusTest::StatusType getStatus() const;

    virtual std::ostream& print(std::ostream& stream, int indent = 0) const;

  private:

    // Status
    NOX::StatusTest::StatusType d_status;

    //! Tolerance required for convergence
    double d_tolerance;

    //! Ostream used to print errors
    NOX::Utils d_utils;

    //! Current residual norm.
    double d_r_norm;

    //! RHS norm.
    double d_b_norm;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_MCSAStatusTest_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_MCSASTATUSTEST_HPP

//---------------------------------------------------------------------------//
// end MCLS_MCSAStatusTest.hpp
//---------------------------------------------------------------------------//
