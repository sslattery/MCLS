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
 * \file MCLS_MCSAStatusTest_impl.hpp
 * \author Stuart R. Slattery
 * \brief MCSA NOX status test.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MCSASTATUSTEST_IMPL_HPP
#define MCLS_MCSASTATUSTEST_IMPL_HPP

#include "MCLS_MCSAModelEvaluator.hpp"
#include "MCLS_VectorTraits.hpp"

#include <NOX_Common.H>
#include <NOX_Abstract_Vector.H>
#include <NOX_Abstract_Group.H>
#include <NOX_Solver_Generic.H>
#include <NOX_Thyra.H>
#include <NOX_Utils.H>
#include <Teuchos_Assert.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_as.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
template<class Vector, class Matrix>
MCSAStatusTest<Vector,Matrix>::MCSAStatusTest( double itolerance, 
					       const NOX::Utils* u ) 
    : tolerance( d_tolerance )
{
    if (u != NULL)
	d_utils = *u;
}

//---------------------------------------------------------------------------//
template<class Vector, class Matrix>
NOX::StatusTest::StatusType MCSAStatusTest<Vector,Matrix>::
checkStatus(const NOX::Solver::Generic& problem,
            NOX::StatusTest::CheckType checkType)
{
    const NOX::Thyra::Group& group
	= dynamic_cast< const NOX::Thyra::Group& >(problem.getSolutionGroup());

    Teuchos::RCP<const MCSAModelEvaluator<Vector,Matrix> > model = 
	Teuchos::rcp_dynamic_cast<const MCSAModelEvaluator<Vector,Matrix> >(group.getModel());

    // On initial iteration, compute initial RHS norm.
    if (problem.getNumIterations() == 0) 
    {
	d_b_norm = VectorTraits<Vector>::norm2( *model->getRHS() );
    }

    d_r_norm = VectorTraits<Vector>::norm2( *model->getPrecResidual(group.getX()) );

    if (checkType == NOX::StatusTest::None)
    {
	d_status = NOX::StatusTest::Unevaluated;
    }
    else
    {
	d_status = (d_r_norm < tolerance * d_b_norm) ? 
		   NOX::StatusTest::Converged : NOX::StatusTest::Unconverged;
    }

    return d_status;
}

//---------------------------------------------------------------------------//
template<class Vector, class Matrix>
NOX::StatusTest::StatusType 
MCSAStatusTest<Vector,Matrix>::getStatus() const
{
    return d_status;
}

//---------------------------------------------------------------------------//
template<class Vector, class Matrix>
std::ostream& MCSAStatusTest<Vector,Matrix>::
print(std::ostream& stream, int indent) const
{
    for (int j = 0; j < indent; j ++)
	stream << ' ';
    stream << d_status;
    stream << "MCSA |r|_2 / |b|_2 = " 
	   << NOX::Utils::sciformat(d_r_norm/d_b_norm,3);
    stream << " < " << NOX::Utils::sciformat(tolerance);

    return stream;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_MCSASTATUSTEST_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_MCSAStatusTest_impl.hpp
//---------------------------------------------------------------------------//
