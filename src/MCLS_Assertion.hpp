//---------------------------------------------------------------------------//
/*!
 * \file   MCLS_Assertion.hpp
 * \author Stuart Slattery
 * \brief  Assertions and Design-by-Contract for error handling.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ASSERTION_HPP
#define MCLS_ASSERTION_HPP

#include <stdexcept>
#include <string>

#include "MCLS_config.hpp"

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Base class for MCLS assertions. This structure is heavily based on
 * that in Nemesis developed by Tom Evans. We derive from std::logic_error
 * here as the DBC checks that utilize this class are meant to find errors
 * that can be prevented before runtime.
 */
//---------------------------------------------------------------------------//
class Assertion : public std::logic_error
{
  public:

    /*! 
     * \brief Default constructor.
     *
     * \param msg Error message.
     */
    Assertion( const std::string& msg )
	: std::logic_error( msg )
    { /* ... */ }

    /*! 
     * \brief DBC constructor.
     *
     * \param cond A string containing the assertion condition that failed.
     *
     * \param field A string containing the file name in which the assertion
     * failed. 
     *
     * \param line The line number at which the assertion failed.
     */
    Assertion( const std::string& cond, const std::string& file, 
	       const int line )
	: std::logic_error( generate_output( cond, file, line ) )
    { /* ... */ }

    //! Destructor.
    virtual ~Assertion() throw()
    { /* ... */ }

  private:

    // Build an assertion output from advanced constructor arguments.
    std::string generate_output( const std::string& cond, 
				 const std::string& file, 
				 const int line ) const;
};

//---------------------------------------------------------------------------//
// Throw functions.
//---------------------------------------------------------------------------//
// Throw a MCLS::Assertion.
void throwAssertion( const std::string& cond, const std::string& file,
		     const int line );

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Design-by-Contract macros.
//---------------------------------------------------------------------------//
/*!
  \page MCLS Design-by-Contract.
 
  Design-by-Contract (DBC) functionality is provided to verify function
  preconditions, postconditions, and invariants. These checks are separated
  from the debug build and can be activated for both release and debug
  builds. They can be activated by setting the following in a CMake
  configure:
 
  -D MCLS_ENABLE_DBC:BOOL=ON
 
  By default, DBC is deactivated. Although they will require additional
  computational overhead, these checks provide a mechanism for veryifing
  library input arguments. Note that the bounds-checking functionality used
  within the MCLS is only provided by a debug build.
 
  In addition, remember is provided to store values used only for DBC
  checks and no other place in executed code.

  Separate from the DBC build, testAssertion can be used at any time verify a
  conditional. This should be used instead of the standard cassert.
 */

#if HAVE_CHIMERA_DBC

#define testPrecondition(c) \
    if (!(c)) MCLS::throwAssertion( #c, __FILE__, __LINE__ )
#define testPostcondition(c) \
    if (!(c)) MCLS::throwAssertion( #c, __FILE__, __LINE__ )
#define testInvariant(c) \
    if (!(c)) MCLS::throwAssertion( #c, __FILE__, __LINE__ )
#define remember(c) c

#else

#define testPrecondition(c)
#define testPostcondition(c)
#define testInvariant(c)
#define remember(c)

#endif


#define testAssertion(c) \
    if (!(c)) MCLS::throwAssertion( #c, __FILE__, __LINE__ )

//---------------------------------------------------------------------------//

#endif // end MCLS_ASSERTION_HPP

//---------------------------------------------------------------------------//
// end MCLS_Assertion.hpp
//---------------------------------------------------------------------------//

