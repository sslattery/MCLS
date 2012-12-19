//---------------------------------------------------------------------------//
/*!
 * \file   MCLS_Assertion.cpp
 * \author Stuart Slattery
 * \brief  Assertions for error handling and Design-by-Contract.
 */
//---------------------------------------------------------------------------//

#include <sstream>

#include "MCLS_Assertion.hpp"

namespace MCLS
{
//---------------------------------------------------------------------------//
// Assertion functions.
//---------------------------------------------------------------------------//
/*!
 * \brief Build an assertion output from advanced constructor arguments.
 *
 * \param cond A string containing the assertion condition that failed.
 *
 * \param field A string containing the file name in which the assertion
 * failed. 
 *
 * \param line The line number at which the assertion failed.
 *
 * \return Assertion output.
 */
std::string Assertion::generate_output( 
    const std::string& cond, const std::string& file, const int line ) const
{
    std::ostringstream output;
    output << "MCLS Assertion: " << cond << ", failed in " << file
	   << ", line " << line  << "." << std::endl;
    return output.str();
}

//---------------------------------------------------------------------------//
// Throw functions.
//---------------------------------------------------------------------------//
/*!
 * \brief Throw a MCLS::Assertion.
 *
 * \param cond A string containing the assertion condition that failed.
 *
 * \param field A string containing the file name in which the assertion
 * failed. 
 *
 * \param line The line number at which the assertion failed.
 */
void throwAssertion( const std::string& cond, const std::string& file,
		     const int line )
{
    throw Assertion( cond, file, line );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_Assertion.cpp
//---------------------------------------------------------------------------//
